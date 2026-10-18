from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

OUTCOMES = ("correct", "wrong", "over_abstain", "proper_abstain", "fabricated", "error")
SHOULD_ABSTAIN = ("F3", "F5")  # plus F6 rows whose label is "abstain"


def cell(onehot: dict) -> str:
    return next(k for k, v in onehot.items() if v)


def load(log_dirs: list[str]) -> list[dict]:
    rows = []
    for d in log_dirs:
        for f in sorted(Path(d).glob("*.eval")):
            log = read_eval_log(str(f))
            if log.status != "success" or not log.samples or log.eval.task != "receipts":
                continue
            system = (log.eval.task_args or {}).get("system", "?")
            model = "—" if system == "regex" else log.eval.model.removeprefix("openrouter/")
            for s in log.samples:
                sc = s.scores or {}
                if "outcome" not in sc:
                    continue
                md = s.metadata
                rows.append({
                    "run": Path(d).name, "system": system, "model": model, "id": s.id, "epoch": s.epoch,
                    "family": md["family"], "base_id": md["base_id"], "expected": md["expected"],
                    "outcome": cell(sc["outcome"].value),
                    "receipt": cell(sc["receipt_fidelity"].value) if "receipt_fidelity" in sc else "na",
                    "parser": cell(sc["parse_path"].value) if "parse_path" in sc else "na",
                    "assumption": cell(sc["assumption_fidelity"].value) if "assumption_fidelity" in sc else "na",
                })
    return rows


def rate_ci(rows: list[dict], hit) -> tuple[float, float, int]:  # type: ignore[no-untyped-def]
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan"), 0
    ys = [1.0 if hit(r) else 0.0 for r in rows]
    m = sum(ys) / n
    groups: dict[str, float] = defaultdict(float)
    for r, y in zip(rows, ys, strict=True):
        groups[r["base_id"]] += y - m
    g = len(groups)
    se = math.sqrt(g / (g - 1) * sum(v * v for v in groups.values())) / n if g > 1 else float("nan")
    return m, 1.96 * se, n


def fmt(m: float, h: float, n: int) -> str:
    if n == 0 or math.isnan(m):
        return "—"
    return f"{m:.2f} ± {h:.2f}" if not math.isnan(h) else f"{m:.2f}"


def should_abstain(r: dict) -> bool:
    return r["expected"] == "abstain"


def headline(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["run"], r["system"], r["model"]) for r in rows})
    for run, system, model in keys:
        rs = [r for r in rows if (r["run"], r["system"], r["model"]) == (run, system, model)]
        ans = [r for r in rs if r["expected"] == "answer"]
        abst = [r for r in rs if should_abstain(r)]
        answered = [r for r in rs if r["outcome"] in ("correct", "wrong", "fabricated")]
        gw = [r for r in rs if r["receipt"] != "na"]
        f7 = [r for r in rs if r["family"] == "F7"]
        out.append({
            "run": run, "system": system, "model": model, "n": len(rs),
            "right behaviour": rate_ci(rs, lambda r: r["outcome"] in ("correct", "proper_abstain")),
            "fabricated (of should-abstain)": rate_ci(abst, lambda r: r["outcome"] == "fabricated"),
            "wrong (of should-answer)": rate_ci(ans, lambda r: r["outcome"] == "wrong"),
            "over-abstain (of should-answer)": rate_ci(ans, lambda r: r["outcome"] == "over_abstain"),
            "selective accuracy": rate_ci(answered, lambda r: r["outcome"] == "correct"),
            "error": rate_ci(rs, lambda r: r["outcome"] == "error"),
            "forged receipt": rate_ci(gw, lambda r: r["receipt"] == "forged"),
            "forged AND answered": rate_ci(gw, lambda r: r["receipt"] == "forged" and r["outcome"] in ("correct", "wrong", "fabricated")),
            "LLM parse rate": rate_ci([r for r in gw if r["parser"] in ("llm", "regex-fallback")], lambda r: r["parser"] == "llm"),
            "F7 substituted": rate_ci(f7, lambda r: r["assumption"] == "substituted"),
        })
    return out


def by_family(rows: list[dict]) -> tuple[list[str], dict[tuple, dict[str, str]]]:
    fams = sorted({r["family"] for r in rows})
    table: dict[tuple, dict[str, str]] = {}
    for key in sorted({(r["run"], r["system"], r["model"]) for r in rows}):
        rs = [r for r in rows if (r["run"], r["system"], r["model"]) == key]
        table[key] = {f: fmt(*rate_ci([r for r in rs if r["family"] == f],
                                      lambda r: r["outcome"] in ("correct", "proper_abstain"))) for f in fams}
    return fams, table


def paired(rows: list[dict], a: tuple[str, str, str], b: tuple[str, str, str]) -> str:
    from scipy.stats import binomtest

    def per_item(key: tuple[str, str, str]) -> dict[str, bool]:
        acc: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            if (r["run"], r["system"], r["model"]) == key:
                acc[r["id"]].append(r["outcome"] in ("correct", "proper_abstain"))
        return {i: sum(v) * 2 > len(v) for i, v in acc.items()}

    pa, pb = per_item(a), per_item(b)
    common = sorted(set(pa) & set(pb))
    a_only = sum(pa[i] and not pb[i] for i in common)
    b_only = sum(pb[i] and not pa[i] for i in common)
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return f"{len(common)} shared items: A right & B wrong {a_only}, B right & A wrong {b_only}, exact McNemar p = {p:.3g}"


def main(dirs: list[str]) -> None:
    rows = load(dirs)
    if not rows:
        sys.exit(f"no receipts logs in {dirs}")
    out = Path("results")
    out.mkdir(exist_ok=True)
    head = headline(rows)
    cols = [k for k in head[0] if k not in ("run", "system", "model", "n")]

    lines = ["# Results", "", f"{len(rows)} scored samples from {', '.join(dirs)}. "
             "Each cell: rate ± 95% half-width (clustered by base question).", ""]
    lines += ["| run | system | model | n | " + " | ".join(cols) + " |",
              "|" + "---|" * (4 + len(cols))]
    for h in head:
        lines.append(f"| {h['run']} | {h['system']} | {h['model']} | {h['n']} | "
                     + " | ".join(fmt(*h[c]) for c in cols) + " |")

    fams, fam_table = by_family(rows)
    lines += ["", "## Right behaviour by family (correct or proper abstain)", "",
              "| run | system | model | " + " | ".join(fams) + " |", "|" + "---|" * (3 + len(fams))]
    for (run, system, model), cells in fam_table.items():
        lines.append(f"| {run} | {system} | {model} | " + " | ".join(cells[f] for f in fams) + " |")

    lines += ["", "## Paired comparisons (same items, exact McNemar)", ""]
    keys = sorted({(r["run"], r["system"], r["model"]) for r in rows})
    for run, system, model in keys:
        if system != "groundwork":
            continue
        for other in ("bare", "bare_abstain"):
            # the bare controls don't use the agent, so they are run once (in "before")
            same_run = [k for k in keys if k[1:] == (other, model) and k[0] == run]
            any_run = [k for k in keys if k[1:] == (other, model)]
            b = (same_run or any_run or [None])[0]
            if b is not None:
                lines.append(f"- {run} · {model}: groundwork vs {other} ({b[0]}): {paired(rows, (run, system, model), b)}")
    for run_a, run_b in [(dirs[0], dirs[-1])] if len(dirs) > 1 else []:
        ra, rb = Path(run_a).name, Path(run_b).name
        for system, model in sorted({(r["system"], r["model"]) for r in rows}):
            a, b = (rb, system, model), (ra, system, model)
            if a in keys and b in keys:
                lines.append(f"- {system} · {model}: {rb} vs {ra}: {paired(rows, a, b)}")

    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    with (out / "samples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or ["logs/before", "logs/after"])
