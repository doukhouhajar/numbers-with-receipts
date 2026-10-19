from __future__ import annotations

import logging
import math
import re

from groundwork.models import Check, Derived, Quantity
from groundwork.tools import dimension_of

log = logging.getLogger(__name__)

def _recipes():
    from groundwork.agent import RECIPES  # agent imports us, avoid the cycle
    return RECIPES


def reexecute(target: str, quantities: dict[str, Quantity]) -> tuple[float | None, str]:
    from groundwork import domain

    q = quantities.get(target)
    if q is None or not isinstance(q.provenance, Derived):
        return None, f"{target} is not a Derived quantity"
    if not q.provenance.method.startswith("analytic:"):
        return None, f"not cheaply re-executable ({q.provenance.method})"

    recipe = _recipes().get(target)
    if recipe is None:
        return None, f"no recipe registered for {target!r}"
    fn_name, leaf_kinds = recipe   # leaf_kinds is the {name: kind} dict
    fn = getattr(domain, fn_name, None)
    if fn is None:
        return None, f"domain has no callable {fn_name!r}"

    try:
        args = [quantities[n].value for n in leaf_kinds]   # dict iterates in signature order
        redo = fn(*args)
    except Exception as exc:
        return None, f"re-execution raised {exc!r}"
    if redo.value is None:
        return None, "re-execution produced Unknown"
    argstr = ", ".join(f"{n}={quantities[n].value!r}" for n in leaf_kinds)
    return redo.value, f"{fn_name}({argstr}) = {redo.value}"


def check_faithfulness(target: str, quantities: dict[str, Quantity], result: Quantity | None) -> Check:
    if result is None or result.value is None:
        return Check(name="faithfulness", status="na", detail="no grounded result to re-execute")
    recomputed, detail = reexecute(target, quantities)
    if recomputed is None:
        return Check(name="faithfulness", status="na", detail=detail)
    # relative: v1 was absolute 1e-9 below 1, so 3e-14 "matched" 3e-15
    ok = math.isclose(recomputed, result.value, rel_tol=1e-9, abs_tol=0.0)
    return Check(
        name="faithfulness",
        status="pass" if ok else "fail",
        severity="blocking",
        detail=f"re-exec {recomputed} vs stated {result.value} — {'match' if ok else 'MISMATCH'}",
    )

def check_dimensional(target: str, quantities: dict[str, Quantity], result: Quantity) -> Check:
    from groundwork import domain

    band = domain.band_for(target)
    if band is None:
        return Check(name="dimensional", status="na",
                     detail=f"no declared dimension for {target}")
    actual = dimension_of(result.unit)
    if actual is None:
        return Check(name="dimensional", status="fail", severity="advisory",
                     detail=f"unit {result.unit!r} is not parseable")
    # Pint renders dimensionless as "dimensionless"; normalise both sides.
    exp = band.expected_dimension.replace("dimensionless", "")
    got = actual.replace("dimensionless", "")
    ok = exp == got
    return Check(
        name="dimensional",
        status="pass" if ok else "fail",
        severity="advisory",
        detail=f"{target}: unit {result.unit or '(dimensionless)'} -> {actual}; "
               f"expected {band.expected_dimension}"
               + ("" if ok else " — MISMATCH"),
    )


def check_magnitude(target: str, quantities: dict[str, Quantity], result: Quantity) -> Check:
    from groundwork import domain

    band = domain.band_for(target)
    if band is None or result.value is None:
        return Check(name="magnitude", status="na",
                     detail=f"no band declared for {target}")
    v = result.value
    below = band.lo is not None and v < band.lo
    above = band.hi is not None and v > band.hi
    if below or above:
        bound = f"< {band.lo}" if below else f"> {band.hi}"
        return Check(
            name="magnitude", status="fail", severity="advisory",
            detail=f"{target} = {v:g} is {bound} plausible range "
                   f"[{band.lo}, {band.hi}] — {band.rationale}",
        )
    return Check(
        name="magnitude", status="pass", severity="advisory",
        detail=f"{target} = {v:g} within [{band.lo}, {band.hi}]",
    )

def qualtran_physical_qubits(
    logical_qubits: int, toffoli_count: int, code_distance: int
) -> tuple[float | None, str]:
    try:
        import qualtran.surface_code as sc
        from qualtran.resource_counting import GateCounts
    except Exception as exc:
        return None, f"Qualtran unavailable ({type(exc).__name__})"
    try:
        algo = sc.AlgorithmSummary(
            n_algo_qubits=logical_qubits,
            n_logical_gates=GateCounts(toffoli=toffoli_count),
        )
        model = sc.PhysicalCostModel.make_gidney_fowler(data_d=code_distance)
        return float(model.n_phys_qubits(algo)), "qualtran.surface_code.make_gidney_fowler"
    except Exception as exc:
        return None, f"Qualtran API mismatch: {exc}"
    
def check_cross_method(target: str, quantities: dict[str, Quantity], result: Quantity) -> Check:

    if target != "physical_qubits" or result.value is None:
        return Check(name="cross_method", status="na", severity="advisory",
                     detail=f"no second estimator registered for {target}")

    n_logical = quantities.get("logical_qubits")
    d = quantities.get("code_distance")      # Azure reports this in its breakdown
    toff = quantities.get("toffoli_count")   # workload's Toffoli count
    missing = [n for n, q in
               (("logical_qubits", n_logical), ("code_distance", d), ("toffoli_count", toff))
               if q is None or not q.is_grounded]
    if missing:
        return Check(name="cross_method", status="na", severity="advisory",
                     detail=f"cannot cross-check: missing grounded {missing}")

    qt_value, qt_note = qualtran_physical_qubits(
        int(n_logical.value), int(toff.value), int(d.value))
    if qt_value is None:
        return Check(name="cross_method", status="na", severity="advisory",
                     detail=qt_note)

    ratio = max(result.value, qt_value) / min(result.value, qt_value)
    ok = ratio <= 2.0
    return Check(
        name="cross_method",
        status="pass" if ok else "fail",
        severity="advisory",
        detail=(f"Azure={result.value:.4g} vs Qualtran={qt_value:.4g} at d={int(d.value)} "
                f"({ratio:.2f}x, agree if <=2x). "
                f"Note: Qualtran whole-algorithm error exceeds target at this d — "
                f"qubit counts comparable, sizing is a lower bound."),
    )

# read-back: every "given by user" number must appear in the question
# (own number reader, independent of the parser it audits)

_SUP = str.maketrans("⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹", "-+0123456789")
_WORDS = {
    "one in a thousand": 1e-3, "one in ten thousand": 1e-4, "one in a hundred thousand": 1e-5,
    "one in a million": 1e-6, "one in a billion": 1e-9, "one in a trillion": 1e-12,
}
_NUMBER = re.compile(
    r"(?P<mant>(?:(?<![\w.])-)?\d[\d,]*(?:\.\d+)?|(?<![\w.])-?\.\d+)"
    r"(?:\s*[x×*]\s*10\s*\^\s*\(?(?P<exp1>[-+]?\d+)\)?|[eE](?P<exp2>[-+]?\d+))?"
    r"(?P<suffix>\s*(?:%|‰|percent\b|per\s*cent\b))?"
)
_POWER = re.compile(r"(?<![\d.])10\s*\^\s*\(?(?P<exp>[-+]?\d+)\)?(?P<suffix>\s*%)?")


def numbers_in(text: str) -> list[float]:
    s = re.sub(r"([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "^" + m.group(1).translate(_SUP), text)
    s = s.replace("−", "-")
    found: list[float] = []
    for phrase, value in _WORDS.items():
        if phrase in s.lower():
            found.append(value)
    taken: list[tuple[int, int]] = []
    for m in _POWER.finditer(s):
        if m.start() > 0 and s[m.start() - 1] in "x×*" or re.search(r"\d\s*[x×*]\s*$", s[:m.start()]):
            continue  # mantissa × 10^k is handled below
        v = 10.0 ** int(m.group("exp"))
        found.append(v / 100 if m.group("suffix") else v)
        taken.append(m.span())
    for m in _NUMBER.finditer(s):
        if any(a <= m.start() < b for a, b in taken):
            continue
        v = float(m.group("mant").replace(",", ""))
        exp = m.group("exp1") or m.group("exp2")
        if exp:
            v *= 10.0 ** int(exp)
        suffix = (m.group("suffix") or "").strip()
        if suffix:
            v /= 1000 if suffix == "‰" else 100
        found.append(v)
    return found


def check_given_fidelity(question: str, quantities: dict[str, Quantity],
                         extra: dict[str, float] | None = None) -> Check:
    # extra: user values that pick the hardware profile
    given = {n: q.value for n, q in quantities.items()
             if q.provenance.kind == "given" and q.value is not None}
    given.update({k: v for k, v in (extra or {}).items() if v is not None})
    if not given:
        return Check(name="given_fidelity", status="na", severity="blocking", detail="no user-given values")
    written = numbers_in(question)
    unread = {n: v for n, v in given.items()
              if not any(math.isclose(v, w, rel_tol=1e-9, abs_tol=0.0) for w in written)}
    if unread:
        said = ", ".join(f"{w:g}" for w in written) or "none"
        return Check(
            name="given_fidelity", status="fail", severity="blocking",
            detail=("recorded as given by the user but not written in the question: "
                    + ", ".join(f"{n}={v:g}" for n, v in unread.items())
                    + f" (numbers in the question: {said})"))
    return Check(name="given_fidelity", status="pass", severity="blocking",
                 detail="every user-given value appears in the question: "
                        + ", ".join(f"{n}={v:g}" for n, v in given.items()))
