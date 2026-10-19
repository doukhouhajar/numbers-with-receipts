from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from groundwork import domain
from groundwork import verify as verify_mod
from groundwork.models import (
    Answer,
    Assumption,
    Check,
    Derivation,
    Quantity,
    Unknown,
    UserGiven,
)
from groundwork.tools import dimension_of, lookup_constant

log = logging.getLogger(__name__)

MAX_COMPUTE_ATTEMPTS = 2


class ParsedQuery(BaseModel):
    target: Literal["physical_qubits", "runtime", "code_distance", "logical_error_rate", "other"]
    physical_error_rate: float | None = None
    target_logical_error_rate: float | None = None
    code_distance: float | None = None
    # v1: one is_rsa flag, so RSA-4096 got RSA-2048's counts
    algorithm: Literal["rsa", "ecc", "aes", "discrete_log", "other"] | None = None
    key_bits: int | None = None
    # v1 ignored named hardware
    qubit_type: Literal["gate_ns", "gate_us", "majorana", "trapped_ion", "other"] | None = None
    qec_scheme: Literal["surface_code", "floquet_code"] | None = None
    error_budget: float | None = None


_PARSE_PROMPT = """Extract the quantum resource estimation request as JSON. Reply with the JSON object only.

Question: {question}

Return exactly this shape:
{{"target": one of ["physical_qubits","runtime","code_distance","logical_error_rate","other"],
  "physical_error_rate": number or null,
  "target_logical_error_rate": number or null,
  "code_distance": number or null,
  "algorithm": one of ["rsa","ecc","aes","discrete_log","other"] or null,
  "key_bits": integer or null,
  "qubit_type": one of ["gate_ns","gate_us","majorana","trapped_ion","other"] or null,
  "qec_scheme": one of ["surface_code","floquet_code"] or null,
  "error_budget": number or null}}

Rules:
- target is the quantity they ask FOR; use "other" if it is none of the four.
- Copy only numbers the question states. Never estimate, never fill a gap, never use a typical value.
- Write each number as a plain decimal or in e-notation: 0.1% -> 0.001, 10^-3 -> 0.001, "one in a thousand" -> 0.001.
- Keep a number even if it looks invalid (negative, above 1, zero): validation happens later.
- algorithm and key_bits only if a cryptosystem or key size is named (RSA-2048 -> "rsa", 2048).
- qubit_type only if the hardware is named: nanosecond or superconducting gate-based -> "gate_ns", microsecond gate-based -> "gate_us", Majorana -> "majorana", trapped ions -> "trapped_ion"."""


def _extend(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


class AgentState(TypedDict, total=False):
    question: str
    target_name: str  # what we're solving for
    target_unit: str  # expected unit of the answer
    target_dimension: str  # expected dimension
    givens: dict[str, float]  # values the user actually supplied
    quantities: dict[str, Quantity]  # the DAG under construction
    steps: Annotated[list[str], _extend]
    assumptions: Annotated[list[Assumption], _extend]
    verification: dict[str, bool]  # check name -> did not block
    verification_notes: Annotated[list[str], _extend]
    attempts: int
    answer: Answer
    workload: str | None  # key into domain.WORKLOADS
    workload_ask: str  # what was asked when no workload matches
    hardware: dict[str, Any]  # machine named in the question
    hw: dict[str, Any]  # machine the estimator runs on
    blocked: str  # why the plan cannot run
    leaf_kinds: dict[str, str]
    checks: list[Check]
    parser: str  # "llm","regex","regex-fallback"
    parse_note: str  # why the LLM path was not used
    llm_reply: str  # raw LLM reply to the parse prompt


LeafKind = Literal["user_input", "workload", "constant"]

# target -> (method, {input_name: how_to_ground_it})
RECIPES: dict[str, tuple[str, dict[str, LeafKind]]] = {
    "physical_qubits": ("estimator", {"logical_qubits": "workload", "toffoli_count": "workload"}),
    "runtime": ("estimator", {"logical_qubits": "workload", "toffoli_count": "workload"}),
    "code_distance": ("required_code_distance", {
        "physical_error_rate": "user_input",
        "target_logical_error_rate": "user_input",
    }),
    "logical_error_rate": ("logical_error_rate", {
        "physical_error_rate": "user_input",
        "code_distance": "user_input",
    }),
}


# parse ─


def _extract_json(raw: str) -> str:
    # first balanced {...}; v1 only trimmed ``` fences
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return text[start:i + 1]
    raise ValueError("unbalanced JSON object in the reply")


def _target_unit(target: str) -> tuple[str, str]:
    unit = "s" if target == "runtime" else ""
    return unit, dimension_of(unit) or "dimensionless"


def parse(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    conf = (config or {}).get("configurable", {})
    if not conf.get("use_llm", True):
        return {**_parse_regex(state), "parser": "regex", "parse_note": "regex-only run", "llm_reply": ""}

    raw = ""
    try:
        llm = conf.get("llm")
        if llm is None:
            from groundwork.config import default_llm
            llm = default_llm()
        raw = llm(_PARSE_PROMPT.format(question=state["question"]))
        p = ParsedQuery.model_validate_json(_extract_json(raw))
    except Exception as exc:
        note = f"{type(exc).__name__}: {str(exc)[:300]}"
        log.warning("LLM parse failed (%s) — regex fallback", note)
        return {**_parse_regex(state), "parser": "regex-fallback", "parse_note": note,
                "llm_reply": raw[:2000]}

    givens = {k: v for k, v in {
        "physical_error_rate": p.physical_error_rate,
        "target_logical_error_rate": p.target_logical_error_rate,
        "code_distance": p.code_distance,
    }.items() if v is not None}
    workload = domain.find_workload(p.algorithm, p.key_bits) if p.algorithm else None
    ask = f"{p.algorithm}" + (f"-{p.key_bits}" if p.key_bits else "") if p.algorithm and not workload else ""
    unit, dim = _target_unit(p.target)
    return {
        "target_name": p.target, "target_unit": unit, "target_dimension": dim,
        "givens": givens,
        "workload": workload, "workload_ask": ask,
        "hardware": {"qubit_type": p.qubit_type, "qec_scheme": p.qec_scheme,
                     "error_budget": p.error_budget},
        "steps": [f"[llm] Parsed target: {p.target}, givens: {list(givens)}"],
        "attempts": 0,
        "parser": "llm", "parse_note": "", "llm_reply": raw[:2000],
    }


_NUM = r"(-?\d*\.?\d+(?:[eE][-+]?\d+)?)"  # v1 swallowed a final period
_HARDWARE_WORDS = ("majorana", "floquet", "trapped", " ion", "microsecond", "nanosecond",
                   "gate-based", "superconducting", "error budget", "µs", " ns ")


def _parse_regex(state: AgentState) -> dict[str, Any]:
    # reference reader, still percent-blind on purpose: read-back protects the receipts
    q = state["question"].lower()
    givens: dict[str, float] = {}

    if m := re.search(r"physical error rate\s*(?:of\s*|is\s*)?" + _NUM, q):
        givens["physical_error_rate"] = float(m.group(1))
    if m := re.search(r"logical error rate\s*(?:of\s*|at\s*)?" + _NUM, q):
        givens["target_logical_error_rate"] = float(m.group(1))
    if m := re.search(r"distance[-\s]*(?:of\s*)?(\d+)", q):
        givens["code_distance"] = float(m.group(1))

    if re.search(r"what(?: is|'s) the logical error rate", q):
        target = "logical_error_rate"
    elif "distance" in q:
        target = "code_distance"
    elif "runtime" in q or "how long" in q:
        target = "runtime"
    elif "qubits" in q:
        target = "physical_qubits"
    else:
        target = "other"

    workload, ask = None, ""
    if re.search(r"rsa[- ]?2048|2048-bit rsa", q):
        workload = "rsa2048_gidney2025"
    elif any(w in q for w in ("rsa", "shor", "factor", "ecc", "aes", "grover", "logarithm")):
        ask = "the named algorithm or key size"

    unparsed = [w.strip() for w in _HARDWARE_WORDS if w in f" {q} "]
    unit, dim = _target_unit(target)
    return {
        "target_name": target, "target_unit": unit, "target_dimension": dim,
        "givens": givens,
        "workload": workload, "workload_ask": ask,
        "hardware": {"unparsed": ", ".join(unparsed)} if unparsed else {},
        "steps": [f"Parsed target: {target}, givens: {list(givens)}"],
        "attempts": 0,
    }


# plan
def _resolve_hardware(state: AgentState) -> tuple[dict[str, Any], str | None]:
    # named machine, or declared default when none is named; never a silent substitute
    hw = state.get("hardware") or {}
    if hw.get("unparsed"):
        return {}, f"the question names hardware ({hw['unparsed']}) that this reader cannot map to a qubit profile"
    qtype = hw.get("qubit_type")
    p = (state.get("givens") or {}).get("physical_error_rate")
    qec = hw.get("qec_scheme") or "surface_code"
    budget = hw.get("error_budget") if hw.get("error_budget") is not None else 0.01
    if not 0 < budget < 1:
        return {}, f"error budget {budget:g} must lie in (0, 1)"
    if qtype is None and p is None:
        profile, default = domain.DEFAULT_QUBIT_PROFILE, True
    else:
        profile = domain.match_qubit_profile(qtype, p)
        if profile is None:
            known = ", ".join(sorted(domain.QUBIT_PROFILES))
            return {}, (f"no Azure qubit profile for qubit type {qtype or 'unspecified'} at physical "
                        f"error rate {'unspecified' if p is None else f'{p:g}'} (known: {known})")
        default = False
    if qec == "floquet_code" and not profile.startswith("qubit_maj"):
        return {}, "a Floquet code needs Majorana qubits in Azure's model; name the qubit type and error rate"
    return {"qubit_profile": profile, "qec_scheme": qec, "error_budget": budget,
            "default_profile": default, "physical_error_rate": p}, None


def plan(state: AgentState) -> dict[str, Any]:
    target = state["target_name"]
    recipe = RECIPES.get(target)
    if recipe is None:
        return {
            "quantities": {target: Quantity(
                name=target, provenance=Unknown(reason=f"no recipe registered for {target!r}"))},
            "steps": [f"No capability produces {target!r}"],
        }

    method, leaf_kinds = recipe
    quantities: dict[str, Quantity] = {
        name: Quantity(name=name, provenance=Unknown(reason="pending grounding"))
        for name in leaf_kinds
    }
    quantities[target] = Quantity(
        name=target, unit=state["target_unit"], dimension=state["target_dimension"],
        provenance=Unknown(reason="pending computation"),
    )
    out: dict[str, Any] = {"quantities": quantities, "leaf_kinds": dict(leaf_kinds),
                           "steps": [f"Planned: {target} <- {method}({', '.join(leaf_kinds)})"]}

    if method == "estimator":
        hw, problem = _resolve_hardware(state)
        if problem:
            quantities[target] = Quantity(name=target, provenance=Unknown(reason=problem))
            out.update(blocked=problem, steps=[f"Cannot plan {target}: {problem}"])
            return out
        prof = domain.QUBIT_PROFILES[hw["qubit_profile"]]
        how = "default: the question names no hardware" if hw["default_profile"] else "as the question specifies"
        out["hw"] = hw
        out["assumptions"] = [
            Assumption(text=f"Azure qubit profile {prof.name} ({prof.description}), {how}",
                       impact="dominates the physical qubit count and runtime"),
            Assumption(text=f"{hw['qec_scheme']}, error budget {hw['error_budget']:g}",
                       impact="sets the code distance"),
        ]
    else:
        out["assumptions"] = [Assumption(
            text="surface code, Azure QRE convention P_L = 0.03 * (p / 0.01)^((d+1)/2)",
            impact="another prefactor or threshold shifts the answer")]
    return out


# ground 


def ground(state: AgentState) -> dict[str, Any]:
    givens = state.get("givens", {})
    leaf_kinds = state.get("leaf_kinds", {})
    quantities = dict(state["quantities"])
    notes: list[str] = []

    workload = state.get("workload")
    workload_counts = domain.workload_counts(workload) if workload else {}

    for name, kind in leaf_kinds.items():
        q = quantities[name]

        if name in givens:
            quantities[name] = Quantity(
                name=name, value=givens[name], unit=q.unit,
                dimension=q.dimension, provenance=UserGiven())
            notes.append(f"Grounded {name} from user input")
            continue

        if kind == "user_input":
            quantities[name] = Quantity(name=name, provenance=Unknown(
                reason=f"you did not specify {name.replace('_', ' ')} — please provide it"))
            notes.append(f"Missing user input: {name}")

        elif kind == "workload":
            if name in workload_counts and workload_counts[name].is_grounded:
                quantities[name] = workload_counts[name]
                notes.append(f"Grounded {name} from published workload counts")
            else:
                ask = state.get("workload_ask")
                known = ", ".join(sorted(domain.WORKLOADS))
                why = (f"no published logical counts for {ask} (known: {known})" if ask
                       else f"no algorithm with published counts was named (known: {known})")
                quantities[name] = Quantity(name=name, provenance=Unknown(reason=why))
                notes.append(f"Could not ground {name} from workloads")

        elif kind == "constant":
            found = lookup_constant(name)
            quantities[name] = found
            notes.append(f"Grounded {name} from constants" if found.is_grounded
                         else f"No sourced constant for {name}")

    return {"quantities": quantities, "steps": notes}


# compute ──────────────────────────────────────────────────────────────────


def compute(state: AgentState) -> dict[str, Any]:
    target = state["target_name"]
    quantities = dict(state["quantities"])
    attempts = state.get("attempts", 0) + 1

    recipe = RECIPES.get(target)
    if recipe is None or state.get("blocked"):
        return {"attempts": attempts}

    method, required = recipe
    missing = [n for n in required if not quantities.get(n, Quantity(name=n, provenance=Unknown(reason="absent"))).is_grounded]
    if missing:
        quantities[target] = Quantity(
            name=target,
            provenance=Unknown(reason=f"inputs not grounded: {', '.join(missing)}"),
        )
        return {
            "quantities": quantities,
            "attempts": attempts,
            "steps": [f"Cannot compute {target}: missing {missing}"],
        }

    args = {n: quantities[n].value for n in required}
    try:
        if method == "estimator":
            hw = state["hw"]
            results = domain.estimate_from_logical_counts(
                logical_qubits=int(args["logical_qubits"]),
                toffoli_count=int(args["toffoli_count"]),
                qubit_profile=hw["qubit_profile"],
                qec_scheme=hw["qec_scheme"],
                error_budget=hw["error_budget"],
            )
            # splice in every reported quantity; they're all receipted
            for name, q in results.items():
                if name != target and q.is_grounded:
                    quantities[name] = q
            produced = results.get(target)
            if produced is None:
                produced = Quantity(
                    name=target,
                    provenance=Unknown(reason=f"estimator did not report {target!r}"),
                )
        else:
            fn = getattr(domain, method)
            produced = fn(*[args[n] for n in required])
    except domain.EstimatorError as exc:
        log.warning("estimator failed: %s", exc)
        produced = Quantity(name=target, provenance=Unknown(reason=f"estimator failed: {exc}"))

    quantities[target] = produced
    return {
        "quantities": quantities,
        "attempts": attempts,
        "steps": [f"Computed {target} via {method}"],
    }


# verify 


def verify(state: AgentState) -> dict[str, Any]:
    target = state["target_name"]
    quantities = state.get("quantities", {})
    result = quantities.get(target)

    hw_p = (state.get("hw") or {}).get("physical_error_rate")
    checks: list[Check] = [verify_mod.check_given_fidelity(
        state["question"], quantities, extra={"physical_error_rate (hardware)": hw_p} if hw_p is not None else None)]
    if result is not None and result.is_grounded:
        checks.append(verify_mod.check_faithfulness(target, quantities, result))
        checks.append(verify_mod.check_dimensional(target, quantities, result))
        checks.append(verify_mod.check_magnitude(target, quantities, result))
        checks.append(verify_mod.check_cross_method(target, quantities, result))

    return {
        "checks": checks,
        "verification": {c.name: not c.blocks for c in checks},
        "verification_notes": [f"{c.name}: {c.status} — {c.detail}" for c in checks],
    }


# decide 


def decide(state: AgentState) -> dict[str, Any]:
    target = state["target_name"]
    derivation = Derivation(
        target=target,
        quantities=state["quantities"],
        steps=state.get("steps", []),
        assumptions=state.get("assumptions", []),
    )
    checks = state.get("checks", [])
    blocking_failed = [c for c in checks if c.blocks]
    meta = {k: state.get(k, "") for k in ("parser", "parse_note", "llm_reply")}
    meta["hw"] = state.get("hw", {})

    if blocking_failed:  # a failed read-back explains the rest
        why = "; ".join(f"{c.name}: {c.detail}" for c in blocking_failed)
        answer = Answer(status="abstained", question=state["question"], derivation=derivation,
                        checks=checks, abstain_reason=f"Verification failed — {why}", meta=meta)
    elif derivation.ungrounded:
        reasons = "; ".join(f"{q.name}: {q.provenance.reason}" for q in derivation.ungrounded)
        answer = Answer(status="abstained", question=state["question"], derivation=derivation,
                        checks=checks, abstain_reason=f"Cannot answer without: {reasons}", meta=meta)
    else:
        answer = Answer(status="answered", question=state["question"],
                        result=derivation.quantities[target], derivation=derivation,
                        checks=checks, confidence=0.8, meta=meta)
    return {"answer": answer}


# routing

def route_after_verify(state: AgentState) -> Literal["compute", "decide"]:
    # retry only when faithfulness failed; v1 retried on any failure
    blocked = [n for n, ok in state.get("verification", {}).items() if not ok]
    if "faithfulness" in blocked and state.get("attempts", 0) < MAX_COMPUTE_ATTEMPTS:
        log.info("faithfulness failed — retrying compute")
        return "compute"
    return "decide"


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [
        ("parse", parse),
        ("plan", plan),
        ("ground", ground),
        ("compute", compute),
        ("verify", verify),
        ("decide", decide),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "parse")
    g.add_edge("parse", "plan")
    g.add_edge("plan", "ground")
    g.add_edge("ground", "compute")
    g.add_edge("compute", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["compute", "decide"])
    g.add_edge("decide", END)
    return g.compile()


GRAPH = build_graph()


def run(
    question: str,
    callbacks: list[Any] | None = None,
    llm: Callable[[str], str] | None = None,
    use_llm: bool = True,
) -> Answer:
    final = GRAPH.invoke(
        {"question": question},
        config={"callbacks": callbacks or [], "configurable": {"llm": llm, "use_llm": use_llm}},
    )
    return final["answer"]
