from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import pint
from scipy import constants as sp_constants

from groundwork.config import get_settings
from groundwork.models import Constant, Derived, Quantity, Unknown

log = logging.getLogger(__name__)

@lru_cache
def ureg() -> pint.UnitRegistry:
    return pint.UnitRegistry()


def parse_unit(unit: str) -> pint.Unit | None:
    if not unit:
        return ureg().dimensionless
    try:
        return ureg().parse_units(unit)
    except (pint.UndefinedUnitError, pint.DefinitionSyntaxError, AttributeError):
        log.warning("unparseable unit: %r", unit)
        return None


def dimension_of(unit: str) -> str | None:
    u = parse_unit(unit)
    return None if u is None else str(u.dimensionality)


def convert(q: Quantity, to_unit: str) -> Quantity:
    if q.value is None:
        return q
    src, dst = parse_unit(q.unit), parse_unit(to_unit)
    if src is None or dst is None:
        raise ValueError(f"cannot convert {q.unit!r} -> {to_unit!r}")
    converted = (q.value * ureg().Quantity(1, src)).to(dst)
    return q.model_copy(
        update={
            "value": float(converted.magnitude),
            "unit": to_unit,
            "dimension": str(converted.dimensionality),
        }
    )


_CONSTANTS: dict[str, tuple[float, str, str]] = {
    # name: (value, unit, source)
    "speed_of_light": (sp_constants.c, "m/s", "scipy.constants.c (CODATA)"),
    "planck": (sp_constants.h, "J*s", "scipy.constants.h (CODATA)"),
    "boltzmann": (sp_constants.k, "J/K", "scipy.constants.k (CODATA)"),
    # quantum-domain reference values: cite the paper
    "surface_code_threshold": (
        0.01,
        "",
        "Fowler et al., 'Surface codes: Towards practical large-scale quantum "
        "computation', PRA 86, 032324 (2012) — threshold ~1%",
    ),
}


def lookup_constant(name: str) -> Quantity:
    entry = _CONSTANTS.get(name)
    if entry is None:
        return Quantity(
            name=name,
            provenance=Unknown(reason=f"no sourced value available for {name!r}"),
        )
    value, unit, source = entry
    return Quantity(
        name=name,
        value=float(value),
        unit=unit,
        dimension=dimension_of(unit) or "",
        provenance=Constant(source=source),
    )


def register_constant(name: str, value: float, unit: str, source: str) -> None:
    if not source.strip():
        raise ValueError(f"refusing to register {name!r} without a source")
    _CONSTANTS[name] = (value, unit, source)


# sandboxed execution
_PREAMBLE = """\
import json, math
import numpy as np
from scipy import constants
"""

_EPILOGUE = """\
print("__GROUNDWORK__" + json.dumps({"result": float(result)}))
"""


class SandboxError(RuntimeError):
    """Generated code failed to run or produced no usable result"""

# execute code in an isolated E2B sandbox and return its result
def run_code(code: str, timeout: int = 60) -> dict[str, Any]:
    settings = get_settings()
    if not settings.e2b_api_key:
        raise SandboxError("E2B_API_KEY is not set — cannot execute code")

    from e2b_code_interpreter import Sandbox

    full = f"{_PREAMBLE}\n{code}\n{_EPILOGUE}"
    log.debug("sandbox exec:\n%s", full)

    with Sandbox.create(api_key=settings.e2b_api_key) as sandbox:
        execution = sandbox.run_code(full, timeout=timeout)

    if execution.error:
        raise SandboxError(f"{execution.error.name}: {execution.error.value}")

    for line in execution.logs.stdout:
        if "__GROUNDWORK__" in line:
            payload = line.split("__GROUNDWORK__", 1)[1].strip()
            return json.loads(payload)

    raise SandboxError("code ran but never assigned a result")

def compute_quantity(
    name: str,
    code: str,
    inputs: dict[str, Quantity],
    unit: str = "",
    timeout: int = 60,
) -> Quantity:

    ungrounded = [n for n, q in inputs.items() if not q.is_grounded]
    if ungrounded:
        return Quantity(
            name=name,
            provenance=Unknown(reason=f"inputs not grounded: {', '.join(ungrounded)}"),
        )

    bindings = "\n".join(f"{n} = {q.value!r}  # {q.unit or 'dimensionless'}" for n, q in inputs.items())
    result = run_code(f"{bindings}\n{code}", timeout=timeout)

    return Quantity(
        name=name,
        value=float(result["result"]),
        unit=unit,
        dimension=dimension_of(unit) or "",
        provenance=Derived(
            method="sandbox:python",
            inputs=list(inputs),
            code_ref=code.strip(),
        ),
    )

def with_uncertainty(q: Quantity, absolute: float | None = None, relative: float | None = None) -> Quantity:
    if (absolute is None) == (relative is None):
        raise ValueError("give exactly one of absolute= or relative=")
    if q.value is None:
        return q
    sigma = absolute if absolute is not None else abs(q.value) * float(relative)  # type: ignore[arg-type]
    return q.model_copy(update={"uncertainty": float(sigma)})