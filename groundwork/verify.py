from __future__ import annotations

import logging

from groundwork.models import Check, Derived, Quantity

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
    ok = abs(recomputed - result.value) <= 1e-9 * max(1.0, abs(result.value))
    return Check(
        name="faithfulness",
        status="pass" if ok else "fail",
        blocking=True,
        detail=f"re-exec {recomputed} vs stated {result.value} — {'match' if ok else 'MISMATCH'}",
    )