from __future__ import annotations

import contextlib
import dataclasses
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inspect_ai import eval as inspect_eval  # noqa: E402
from tasks import mutation  # noqa: E402

from groundwork import domain  # noqa: E402

# evaluate evaluator 
MUTANTS = {
    "none": "control: no fault",
    "M1_workload_x1000": "published Toffoli count x1000 (a typo in a receipted constant)",
    "M2_t_vs_ccz_slot": "Toffolis sent to the other estimator slot (TCount <-> CczCount)",
    "M3_formula_plus2": "code-distance formula returns d+2, wherever it is called",
    "M4_compute_tampered": "d+2 / P_L x10 on the first call only: compute is wrong, re-execution is clean",
    "M5_prefactor_0.1": "surface-code prefactor 0.1 instead of 0.03",
    "M6_runtime_in_ns": "runtime left in nanoseconds but labelled seconds",
    "M7_reader_swaps_inputs": "the reader swaps the two user numbers",
    "M8_reader_percent_blind": "the reader reads '0.1%' as 0.1",
}

@contextlib.contextmanager
def installed(mutant: str) -> Iterator[None]:
    saved: list[tuple[object, str, object]] = []

    def patch(obj: object, name: str, value: object) -> None:
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    if mutant == "M1_workload_x1000":
        w = domain.WORKLOADS["rsa2048_gidney2025"]
        field = "toffoli_count" if hasattr(w, "toffoli_count") else "t_count"
        patched = dict(domain.WORKLOADS)
        patched[w.key] = dataclasses.replace(w, **{field: getattr(w, field) * 1000})
        patch(domain, "WORKLOADS", patched)

    elif mutant == "M2_t_vs_ccz_slot":
        original = domain._logical_counts_program

        def swapped(*a, **k):  # type: ignore[no-untyped-def]
            prog = original(*a, **k)
            return prog.replace("TCount(", "@@(").replace("CczCount(", "TCount(").replace("@@(", "CczCount(")
        patch(domain, "_logical_counts_program", swapped)

    elif mutant == "M3_formula_plus2":
        original_d = domain.required_code_distance

        def plus2(*a, **k):  # type: ignore[no-untyped-def]
            q = original_d(*a, **k)
            return q if q.value is None else q.model_copy(update={"value": q.value + 2})
        patch(domain, "required_code_distance", plus2)

    elif mutant == "M4_compute_tampered":
        calls: dict[tuple, int] = {}

        def first_call_only(fn, bump):  # type: ignore[no-untyped-def]
            def wrapped(*a, **k):  # type: ignore[no-untyped-def]
                if len(a) != 2 or k:  
                    return fn(*a, **k)
                n = calls[(fn.__name__, a)] = calls.get((fn.__name__, a), 0) + 1
                q = fn(*a, **k)
                return q if (q.value is None or n % 2 == 0) else q.model_copy(update={"value": bump(q.value)})
            return wrapped
        patch(domain, "required_code_distance", first_call_only(domain.required_code_distance, lambda v: v + 2))
        patch(domain, "logical_error_rate", first_call_only(domain.logical_error_rate, lambda v: v * 10))

    elif mutant == "M5_prefactor_0.1":
        bad = dataclasses.replace(domain.SURFACE_CODE, crossing_prefactor=0.1)
        for fn in (domain.required_code_distance, domain.logical_error_rate):
            saved.append((fn, "__defaults__", fn.__defaults__))
            fn.__defaults__ = (bad,)

    elif mutant == "M6_runtime_in_ns":
        original_unpack = domain._unpack_estimate

        def ns(*a, **k):  # type: ignore[no-untyped-def]
            out = original_unpack(*a, **k)
            rt = out.get("runtime")
            if rt is not None and rt.value is not None:
                out["runtime"] = rt.model_copy(update={"value": rt.value * 1e9})
            return out
        patch(domain, "_unpack_estimate", ns)

    try:
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def main() -> None:
    log_dir = Path(__file__).parent.parent / "logs" / "mutation"
    grid: dict[str, dict[str, float]] = {}
    for mutant in MUTANTS:
        with installed(mutant):
            log = inspect_eval(mutation(mutant=mutant), model="mockllm/model", log_dir=str(log_dir),
                               display="none", max_samples=4)[0]
        row: dict[str, float] = {}
        for s in log.results.scores if log.results else []:
            for k, m in s.metrics.items():
                if not k.endswith("_se"):
                    row[f"{s.name}.{k}"] = m.value
        grid[mutant] = row

    cols = [("checks_fired.faithfulness", "faith"), ("checks_fired.dimensional", "dim"),
            ("checks_fired.magnitude", "magn"), ("checks_fired.cross_method", "cross"),
            ("checks_fired.given_fidelity", "given"), ("checks_fired.abstained", "abst"),
            ("checks_fired.detected", "DETECT"), ("outcome.wrong", "wrong"),
            ("receipt_fidelity.forged", "forged")]
    print(f"\n{'mutant':26s}" + "".join(f"{c:>8s}" for _, c in cols))
    for mutant, row in grid.items():
        print(f"{mutant:26s}" + "".join(f"{row.get(k, float('nan')):8.2f}" for k, _ in cols))
    print("\nrates over the mutation set (F1, F2, F6 answerable rows). DETECT = any check failed or the agent abstained.")


if __name__ == "__main__":
    main()
