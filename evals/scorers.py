from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState

OUTCOMES = ("correct", "wrong", "over_abstain", "proper_abstain", "fabricated", "error")

_SUP = str.maketrans("⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹", "-+0123456789")
_SCALE = (("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3))
_TIME = (("year", 3.156e7), ("day", 86400.0), ("hour", 3600.0), ("minute", 60.0))


def parse_number(text: str, as_seconds: bool = False) -> float | None:
    s = text.replace("−", "-").replace("×", "x").replace("·", "x").replace("*", "x")
    s = re.sub(r"([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "^" + m.group(1).translate(_SUP), s)
    s = s.replace("x x", "x")
    m = re.search(r"([-+]?\d[\d,]*\.?\d*)\s*x\s*10\s*\^\s*\(?([-+]?\d+)\)?", s)
    if m:
        value, end = float(m.group(1).replace(",", "")) * 10.0 ** int(m.group(2)), m.end()
    else:
        m = re.search(r"(?<![\d.])10\s*\^\s*\(?([-+]?\d+)\)?", s)
        if m:
            value, end = 10.0 ** int(m.group(1)), m.end()
        else:
            m = re.search(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", s)
            if not m:
                return None
            value, end = float(m.group(0).replace(",", "")), m.end()
    rest = s[end:].strip().lower()
    for word, factor in _SCALE:
        if rest.startswith(word):
            value *= factor
            rest = rest[len(word):].strip()
    if as_seconds:
        for word, factor in _TIME:
            if rest.startswith(word):
                value *= factor
    return value


@dataclass
class Reading:
    kind: str            # "groundwork","bare"
    status: str          # "answered","abstained","error"
    value: float | None
    answer: dict | None  # Groundwork answer
    note: str = ""


def read_completion(text: str, target_name: str | None = None) -> Reading:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict) and "status" in obj:
        if obj["status"] == "error":
            return Reading("groundwork", "error", None, obj, obj.get("error", ""))
        value = (obj.get("result") or {}).get("value")
        return Reading("groundwork", obj["status"], value, obj)

    lines = re.findall(r"ANSWER:\s*`?([^`\n]*)", text or "")
    if not lines:
        return Reading("bare", "error", None, None, "no ANSWER line (contract broken)")
    tail = lines[-1].strip()
    if re.match(r"(?i)(abstain|none|n/?a|unknown|cannot|insufficient)", tail):
        return Reading("bare", "abstained", None, None)
    value = parse_number(tail, as_seconds=(target_name == "runtime"))
    if value is None:
        return Reading("bare", "error", None, None, f"unparseable ANSWER: {tail[:60]}")
    return Reading("bare", "answered", value, None)


def within(value: float | None, md: dict) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if md.get("oracle_value") is not None:
        target, tol = float(md["oracle_value"]), float(md.get("rel_tol") or 0.0)
        if tol == 0.0:
            return abs(value - target) < 1e-9 * max(1.0, abs(target))  # exact: d = 25 is not d = 27
        return abs(value - target) <= tol * abs(target)
    if md.get("range"):
        lo, hi = md["range"]
        return lo <= value <= hi
    return False


def estimator_method(answer: dict | None) -> str:
    prov = ((answer or {}).get("result") or {}).get("provenance") or {}
    return prov.get("method", "")


def honored(answer: dict | None, requested: dict) -> bool:
    method = estimator_method(answer)
    if not requested or not method:
        return False
    for key, want in requested.items():
        if key == "qubit_profile" and f"[{want}/" not in method:
            return False
        if key == "qubit_profile_suffix" and f"{want}/" not in method:
            return False
        if key == "qec_scheme" and f"/{want}/" not in method:
            return False
        if key == "error_budget" and f"budget={want}]" not in method:
            return False
        if key == "hardware":  # no Azure profile corresponds to it
            return False
    return True


def classify(reading: Reading, md: dict) -> str:
    expected = md["expected"]
    if reading.status == "error":
        return "error"
    if expected == "answer":
        if reading.status == "abstained":
            return "over_abstain"
        return "correct" if within(reading.value, md) else "wrong"
    if expected == "abstain":
        return "proper_abstain" if reading.status == "abstained" else "fabricated"
    if expected == "abstain_or_honor":
        if reading.status == "abstained":
            return "proper_abstain"
        ok = reading.kind == "groundwork" and honored(reading.answer, md.get("requested", {}))
        return "correct" if ok and within(reading.value, md) else "wrong"
    raise ValueError(f"unknown expected label {expected!r}")


def read_receipts(answer: dict | None) -> dict[str, float]:
    quantities = ((answer or {}).get("derivation") or {}).get("quantities") or {}
    return {
        name: q["value"]
        for name, q in quantities.items()
        if (q.get("provenance") or {}).get("kind") == "given" and q.get("value") is not None
    }


def _same(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-15)


def receipt_verdict(givens: dict[str, float], expected: dict[str, float]) -> tuple[str, str]:
    forged = [f"{k}={v:g} (user said {expected[k]:g})" if k in expected else f"{k}={v:g} (never said)"
              for k, v in givens.items() if k not in expected or not _same(v, expected[k])]
    if forged:
        return "forged", "; ".join(forged)
    missing = [k for k in expected if k not in givens]
    if missing:
        return "missing", f"not recorded: {missing}"
    return ("faithful", "every given value matches") if givens else ("none", "no user values")


# metrics
def _cluster_se(scores: list[SampleScore], key: str) -> float:
    groups: dict[str, list[float]] = {}
    for s in scores:
        cid = str((s.sample_metadata or {}).get("base_id", s.sample_id))
        groups.setdefault(cid, []).append(float((s.score.value or {}).get(key, 0.0)))
    n = len(scores)
    if n < 2 or len(groups) < 2:
        return float("nan")
    overall = mean(float((s.score.value or {}).get(key, 0.0)) for s in scores)
    g = len(groups)
    resid = sum((sum(v) - len(v) * overall) ** 2 for v in groups.values())
    return math.sqrt(g / (g - 1) * resid) / n


@metric
def rates(keys: tuple[str, ...] = OUTCOMES) -> Metric:
    def compute(scores: list[SampleScore]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in keys:
            vals = [float((s.score.value or {}).get(k, 0.0)) for s in scores]
            out[k] = round(mean(vals), 4) if vals else float("nan")
            out[f"{k}_se"] = round(_cluster_se(scores, k), 4)
        return out

    return compute


@metric
def selective_accuracy() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        c = sum(float((s.score.value or {}).get("correct", 0)) for s in scores)
        w = sum(float((s.score.value or {}).get("wrong", 0)) for s in scores)
        return round(c / (c + w), 4) if c + w else float("nan")

    return compute


# scorers 
def _onehot(keys: tuple[str, ...], hit: str) -> dict[str, int]:
    return {k: int(k == hit) for k in keys}


@scorer(metrics=[rates(), selective_accuracy()])
def outcome() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        reading = read_completion(state.output.completion, md.get("target_name"))
        cell = classify(reading, md)
        return Score(value=_onehot(OUTCOMES, cell), answer=str(reading.value),
                     explanation=f"{cell}: status={reading.status} value={reading.value} {reading.note}".strip())

    return score


RECEIPT_KEYS = ("faithful", "forged", "missing", "none", "na")


@scorer(metrics=[rates(RECEIPT_KEYS)])
def receipt_fidelity() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        reading = read_completion(state.output.completion)
        if reading.kind != "groundwork" or reading.answer is None:
            return Score(value=_onehot(RECEIPT_KEYS, "na"), explanation="no receipts to read")
        verdict, why = receipt_verdict(read_receipts(reading.answer), state.metadata.get("expected_givens", {}))
        return Score(value=_onehot(RECEIPT_KEYS, verdict), explanation=why)

    return score


PARSER_KEYS = ("llm", "regex", "regex-fallback", "na")


@scorer(metrics=[rates(PARSER_KEYS)])
def parse_path() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        reading = read_completion(state.output.completion)
        meta: dict[str, Any] = (reading.answer or {}).get("meta") or {}
        path = meta.get("parser", "na") if reading.kind == "groundwork" else "na"
        return Score(value=_onehot(PARSER_KEYS, path if path in PARSER_KEYS else "na"),
                     explanation=meta.get("parse_note", "") or path)

    return score


ASSUMPTION_KEYS = ("honored", "abstained", "substituted", "na")


@scorer(metrics=[rates(ASSUMPTION_KEYS)])
def assumption_fidelity() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        if md.get("family") != "F7":
            return Score(value=_onehot(ASSUMPTION_KEYS, "na"))
        reading = read_completion(state.output.completion)
        if reading.status != "answered":
            return Score(value=_onehot(ASSUMPTION_KEYS, "abstained"), explanation=reading.status)
        verdict = "honored" if honored(reading.answer, md.get("requested", {})) else "substituted"
        return Score(value=_onehot(ASSUMPTION_KEYS, verdict),
                     explanation=f"asked {md.get('requested')}, ran {estimator_method(reading.answer) or '?'}")

    return score


CHECK_KEYS = ("faithfulness", "dimensional", "magnitude", "cross_method", "given_fidelity", "abstained", "detected")


@scorer(metrics=[rates(CHECK_KEYS)])
def checks_fired() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        reading = read_completion(state.output.completion)
        checks = (reading.answer or {}).get("checks") or []
        fired = {c["name"] for c in checks if c.get("status") == "fail"}
        value = {k: int(k in fired) for k in CHECK_KEYS[:5]}
        value["abstained"] = int(reading.status != "answered")
        value["detected"] = int(bool(fired) or reading.status != "answered")
        return Score(value=value, explanation=f"fired={sorted(fired)} status={reading.status}")

    return score
