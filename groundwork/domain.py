from __future__ import annotations

import logging
import math
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal

from groundwork.models import Constant, Derived, Quantity, Unknown
from groundwork.tools import dimension_of, register_constant

log = logging.getLogger(__name__)

_estimator_queue: queue.Queue[tuple[Callable[[], object], Future]] = queue.Queue()
_estimator_thread: threading.Thread | None = None
_estimator_lock = threading.Lock()  # guards one-time worker startup only


def _estimator_worker() -> None:
    while True:
        fn, future = _estimator_queue.get()
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn())
            except Exception as exc:  # noqa: BLE001 — deliver the error to the caller
                future.set_exception(exc)
        _estimator_queue.task_done()


def _ensure_worker() -> None:
    global _estimator_thread
    with _estimator_lock:
        if _estimator_thread is None:
            _estimator_thread = threading.Thread(
                target=_estimator_worker, name="qsharp-estimator", daemon=True
            )
            _estimator_thread.start()


def _run_on_estimator_thread(fn: Callable[[], object]) -> object:
    _ensure_worker()
    future: Future = Future()
    _estimator_queue.put((fn, future))
    return future.result()  # blocks; re-raises any exception from the worker


FOWLER_2012 = (
    "Fowler, Mariantoni, Martinis, Cleland, 'Surface codes: Towards practical "
    "large-scale quantum computation', Phys. Rev. A 86, 032324 (2012)"
)
BEVERLAND_2022 = (
    "Beverland et al., 'Assessing requirements to scale to practical quantum "
    "advantage' (2022) — methodology behind the Azure Quantum Resource Estimator"
)
GIDNEY_2025 = (
    "Gidney, 'How to factor 2048 bit RSA integers with less than a million noisy "
    "qubits', arXiv:2505.15917 (2025)"
)
GIDNEY_EKERA_2019 = (
    "Gidney & Ekerå, 'How to factor 2048 bit RSA integers in 8 hours using 20 "
    "million noisy qubits' (2019)"
)
AZURE_DOCS = "Azure Quantum Resource Estimator target parameters (Microsoft Learn)"


# qsharp.estimate
@dataclass(frozen=True)
class QubitProfile:
    name: str
    instruction_set: Literal["GateBased", "Majorana"]
    gate_error: float          # one/two-qubit gate or measurement error rate
    operation_time_ns: float   # characteristic operation time
    description: str


QUBIT_PROFILES: dict[str, QubitProfile] = {
    p.name: p
    for p in [
        QubitProfile("qubit_gate_us_e3", "GateBased", 1e-3, 1_000, "gate-based, µs ops, 10⁻³ error"),
        QubitProfile("qubit_gate_us_e4", "GateBased", 1e-4, 1_000, "gate-based, µs ops, 10⁻⁴ error"),
        QubitProfile("qubit_gate_ns_e3", "GateBased", 1e-3, 50, "gate-based, ns ops, 10⁻³ error (superconducting-like)"),
        QubitProfile("qubit_gate_ns_e4", "GateBased", 1e-4, 50, "gate-based, ns ops, 10⁻⁴ error"),
        QubitProfile("qubit_maj_ns_e4", "Majorana", 1e-4, 100, "Majorana, ns ops, 10⁻⁴ error"),
        QubitProfile("qubit_maj_ns_e6", "Majorana", 1e-6, 100, "Majorana, ns ops, 10⁻⁶ error"),
    ]
}

QEC_SCHEMES = ("surface_code", "floquet_code")


@dataclass(frozen=True)
class QECParams:
    name: str = "surface_code"
    threshold: float = 0.01          # p_th
    crossing_prefactor: float = 0.03  # the 'a' in a·(p/p_th)^((d+1)/2)
    source: str = AZURE_DOCS


SURFACE_CODE = QECParams()
# Majorana-based surface code uses a different threshold; see Azure docs table
SURFACE_CODE_MAJORANA = QECParams(threshold=0.0015, crossing_prefactor=0.08)
FLOQUET_CODE = QECParams(name="floquet_code", threshold=0.01, crossing_prefactor=0.07)

@dataclass(frozen=True)
class PlausibleBand:
    lo: float | None
    hi: float | None
    expected_dimension: str
    rationale: str


# Bands are domain knowledge: they say what a physically sane answer looks like
PLAUSIBLE_BANDS: dict[str, PlausibleBand] = {
    "physical_error_rate": PlausibleBand(
        0.0, 1.0, "dimensionless", "a probability lies in [0, 1]"),
    "target_logical_error_rate": PlausibleBand(
        0.0, 1.0, "dimensionless", "a probability lies in [0, 1]"),
    "logical_error_rate": PlausibleBand(
        0.0, 1.0, "dimensionless", "a probability lies in [0, 1]"),
    "code_distance": PlausibleBand(
        3, 1001, "dimensionless",
        "surface-code distance: below 3 is uncorrectable; above ~1000 is "
        "physically implausible (2d^2 > 2e6 physical qubits per logical qubit)"),
    "physical_qubits": PlausibleBand(
        1e2, 1e10, "dimensionless",
        "RSA-scale factoring is 1e5-1e9 physical qubits (Gidney 2019 vs 2025); "
        "1e2-1e10 is a generous sanity envelope"),
    "runtime": PlausibleBand(
        1e-3, 1e12, "[time]",
        "sub-millisecond is implausibly fast for a fault-tolerant algorithm; "
        "1e12 s (~30000 yr) is the far upper sanity bound"),
    "logical_cycle_time": PlausibleBand(
        1e-9, 1e-3, "[time]",
        "surface-code logical cycle: ~1 us (Gidney); 1 ns-1 ms brackets it widely"),
}


def band_for(name: str) -> PlausibleBand | None:
    return PLAUSIBLE_BANDS.get(name)

def _register_domain_constants() -> None:
    register_constant("surface_code_threshold", 0.01, "", FOWLER_2012 + " — threshold ≈ 1%")
    register_constant("surface_code_crossing_prefactor", 0.03, "", AZURE_DOCS)
    register_constant("rsa2048_logical_qubits_gidney2025", 1409, "", GIDNEY_2025 + ", Table 4/5")
    register_constant("rsa2048_physical_qubits_gidney2025", 1e6, "", GIDNEY_2025 + " — under one million")
    register_constant("rsa2048_runtime_gidney2025", 5 * 24 * 3600, "s", GIDNEY_2025 + " — under one week")
    register_constant("rsa2048_physical_qubits_gidney2019", 20e6, "", GIDNEY_EKERA_2019)
    register_constant("rsa2048_runtime_gidney2019", 8 * 3600, "s", GIDNEY_EKERA_2019)
    register_constant("surface_code_cycle_time_gidney", 1e-6, "s", GIDNEY_2025 + " — 1 µs cycle assumption")


_register_domain_constants()


# analytic QEC formulas 
# P_L ≈ a·(p/p_th)^((d+1)/2)
def logical_error_rate(
    physical_error_rate: float,
    code_distance: int,
    qec: QECParams = SURFACE_CODE,
) -> Quantity:
    if physical_error_rate >= qec.threshold:
        return Quantity(
            name="logical_error_rate",
            provenance=Unknown(
                reason=(
                    f"physical error rate {physical_error_rate:.2e} is at or above the "
                    f"{qec.name} threshold {qec.threshold:.2e}; error correction does not "
                    "suppress errors in this regime and the scaling formula does not apply"
                )
            ),
        )
    if code_distance < 1 or code_distance % 2 == 0:
        return Quantity(
            name="logical_error_rate",
            provenance=Unknown(reason=f"code distance must be a positive odd integer, got {code_distance}"),
        )

    p_l = qec.crossing_prefactor * (physical_error_rate / qec.threshold) ** ((code_distance + 1) / 2)
    return Quantity(
        name="logical_error_rate",
        value=float(p_l),
        unit="",
        dimension="dimensionless",
        provenance=Derived(
            method=f"analytic:surface_code_scaling[{qec.name}]",
            inputs=["physical_error_rate", "code_distance"],
            code_ref=f"P_L = {qec.crossing_prefactor} * (p/{qec.threshold})**((d+1)/2)  # {FOWLER_2012}",
        ),
    )

def required_code_distance(
    physical_error_rate: float,
    target_logical_error_rate: float,
    qec: QECParams = SURFACE_CODE,
) -> Quantity:
    if not physical_error_rate > 0:
        return Quantity(name="code_distance", provenance=Unknown(
            reason=f"physical error rate must be positive, got {physical_error_rate:.3g}"))
    if physical_error_rate >= qec.threshold:
        return Quantity(
            name="code_distance",
            provenance=Unknown(
                reason=(
                    f"physical error rate {physical_error_rate:.2e} is not below the "
                    f"{qec.name} threshold {qec.threshold:.2e}; no code distance achieves the target"
                )
            ),
        )
    if not 0 < target_logical_error_rate < 1:
        return Quantity(
            name="code_distance",
            provenance=Unknown(reason="target logical error rate must lie in (0, 1)"),
        )

    ratio = physical_error_rate / qec.threshold
    # a·ratio^((d+1)/2) ≤ target  implique  d ≥ 2·log(target/a)/log(ratio) − 1
    d_real = 2.0 * math.log(target_logical_error_rate / qec.crossing_prefactor) / math.log(ratio) - 1.0
    d = max(1, math.ceil(d_real))
    if d % 2 == 0:
        d += 1

    forward = logical_error_rate(physical_error_rate, d, qec)
    if forward.value is None or forward.value > target_logical_error_rate:
        return Quantity(
            name="code_distance",
            provenance=Unknown(reason="inverted distance failed its own forward check"),
        )

    return Quantity(
        name="code_distance",
        value=float(d),
        unit="",
        dimension="dimensionless",
        provenance=Derived(
            method=f"analytic:invert_surface_code_scaling[{qec.name}]",
            inputs=["physical_error_rate", "target_logical_error_rate"],
            code_ref=(
                f"d = ceil(2*log(target/{qec.crossing_prefactor})/log(p/{qec.threshold}) - 1), "
                f"rounded up to odd; verified forward. # {FOWLER_2012}"
            ),
        ),
    )

# Surface code: 2d² physical qubits per logical qubit (Azure's formula)
def physical_qubits_per_logical(code_distance: int, qec: QECParams = SURFACE_CODE) -> Quantity:
    if qec.name != "surface_code":
        return Quantity(
            name="physical_qubits_per_logical",
            provenance=Unknown(reason=f"no analytic qubit-count formula wired for {qec.name!r}"),
        )
    return Quantity(
        name="physical_qubits_per_logical",
        value=float(2 * code_distance**2),
        unit="",
        dimension="dimensionless",
        provenance=Derived(
            method="analytic:surface_code_footprint",
            inputs=["code_distance"],
            code_ref=f"n_phys = 2 * d**2  # {AZURE_DOCS}",
        ),
    )

# (4·t_2q + 2·t_meas)·d — Azure's gate-based surface-code cycle time
def logical_cycle_time(
    code_distance: int,
    two_qubit_gate_time_ns: float,
    measurement_time_ns: float,
) -> Quantity:
    t_ns = (4 * two_qubit_gate_time_ns + 2 * measurement_time_ns) * code_distance
    return Quantity(
        name="logical_cycle_time",
        value=float(t_ns * 1e-9),
        unit="s",
        dimension=dimension_of("s") or "[time]",
        provenance=Derived(
            method=(
                f"analytic:surface_code_cycle_time"
                f"[t_2q={two_qubit_gate_time_ns}ns,t_meas={measurement_time_ns}ns]"
            ),
            inputs=["code_distance"],
            code_ref=f"t_cycle = (4*t_2q + 2*t_meas) * d  # {AZURE_DOCS}",
        ),
    )


#Azure estimator 

class EstimatorError(RuntimeError):
    """The resource estimator could not produce an estimate"""

# Q# program that reports pre-computed logical counts to the estimator
def _logical_counts_program(
    logical_qubits: int, t_count: int, rotation_count: int = 0, rotation_depth: int = 0
) -> str:
    return f"""
    import Std.ResourceEstimation.*;
    operation Program() : Unit {{
        use qs = Qubit[{logical_qubits}];
        AccountForEstimates(
            [TCount({t_count}), RotationCount({rotation_count}), RotationDepth({rotation_depth})],
            PSSPCLayout(), qs);
    }}
    """


def estimate_from_logical_counts(
    logical_qubits: int,
    t_count: int,
    qubit_profile: str = "qubit_gate_ns_e3",
    qec_scheme: str = "surface_code",
    error_budget: float = 0.01,
    rotation_count: int = 0,
    rotation_depth: int = 0,
) -> dict[str, Quantity]:
    if qubit_profile not in QUBIT_PROFILES:
        raise EstimatorError(f"unknown qubit profile {qubit_profile!r}; choose from {sorted(QUBIT_PROFILES)}")
    if qec_scheme not in QEC_SCHEMES:
        raise EstimatorError(f"unknown QEC scheme {qec_scheme!r}; choose from {QEC_SCHEMES}")
    if not 0 < error_budget < 1:
        raise EstimatorError(f"error budget must lie in (0, 1), got {error_budget}")

    try:
        import qsharp
    except ImportError as exc:  # pragma: no cover
        raise EstimatorError("the `qsharp` package is not installed") from exc

    program = _logical_counts_program(logical_qubits, t_count, rotation_count, rotation_depth)
    params: dict[str, Any] = {
        "qubitParams": {"name": qubit_profile},
        "qecScheme": {"name": qec_scheme},
        "errorBudget": error_budget,
    }

    log.info("qsharp.estimate: %s / %s / budget=%s", qubit_profile, qec_scheme, error_budget)

    def _do_estimate() -> Any:
        qsharp.init()
        qsharp.eval(program)
        return qsharp.estimate("Program()", params=params)

    try:
        result = _run_on_estimator_thread(_do_estimate)
    except Exception as exc:
        raise EstimatorError(f"resource estimation failed: {exc}") from exc

    return _unpack_estimate(result, qubit_profile, qec_scheme, error_budget)


def _unpack_estimate(
    result: Any, qubit_profile: str, qec_scheme: str, error_budget: float
) -> dict[str, Quantity]:
    phys = result.get("physicalCounts", {}) if hasattr(result, "get") else {}
    breakdown = phys.get("breakdown", {})
    logical = result.get("logicalQubit", {}) if hasattr(result, "get") else {}  
    method = f"qsharp.estimate[{qubit_profile}/{qec_scheme}/budget={error_budget}]"
    inputs = ["logical_qubits", "t_count"]

    def q(name: str, raw: Any, unit: str = "") -> Quantity:
        if raw is None:
            return Quantity(name=name, provenance=Unknown(reason=f"estimator did not report {name!r}"))
        return Quantity(
            name=name,
            value=float(raw),
            unit=unit,
            dimension=dimension_of(unit) or "",
            provenance=Derived(method=method, inputs=inputs, code_ref=f"result.physicalCounts → {name}"),
        )

    runtime_ns = phys.get("runtime")
    return {
        "physical_qubits": q("physical_qubits", phys.get("physicalQubits")),
        "runtime": q("runtime", None if runtime_ns is None else float(runtime_ns) * 1e-9, unit="s"),
        "algorithmic_logical_qubits": q("algorithmic_logical_qubits", breakdown.get("algorithmicLogicalQubits")),
        "logical_depth": q("logical_depth", breakdown.get("logicalDepth")),
        "code_distance": q("code_distance", logical.get("codeDistance")),
        "num_t_states": q("num_t_states", breakdown.get("numTstates")),
        "num_t_factories": q("num_t_factories", breakdown.get("numTfactories")),
    }


# known algorithm workloads 
@dataclass(frozen=True)
class Workload:
    key: str
    description: str
    logical_qubits: int
    t_count: int
    source: str


WORKLOADS: dict[str, Workload] = {
    w.key: w
    for w in [
        Workload("rsa2048_gidney2025", "Factor RSA-2048 (approximate residue arithmetic)",
                 1409, int(6.5e9), GIDNEY_2025 + " — 1409 logical qubits, ~6.5e9 Toffoli"),
        Workload("rsa2048_gidney2019", "Factor RSA-2048 (Ekerå–Håstad, 2019 construction)",
                 6189, int(2.7e12), GIDNEY_EKERA_2019),
    ]
}


def workload_counts(key: str) -> dict[str, Quantity]:
    w = WORKLOADS.get(key)
    if w is None:
        return {
            "logical_qubits": Quantity(name="logical_qubits", provenance=Unknown(
                reason=f"no published logical counts for workload {key!r}; "
                       f"known: {sorted(WORKLOADS)}")),
        }
    return {
        "logical_qubits": Quantity(name="logical_qubits", value=float(w.logical_qubits),
                                   provenance=Constant(source=w.source)),
        "t_count": Quantity(name="t_count", value=float(w.t_count),
                            provenance=Constant(source=w.source)),
    }


CAPABILITIES: dict[str, str] = {
    "logical_error_rate": "P_L from physical error rate and code distance (analytic)",
    "required_code_distance": "smallest odd code distance meeting a target logical error rate (analytic)",
    "physical_qubits_per_logical": "surface-code footprint 2d² (analytic)",
    "logical_cycle_time": "surface-code logical cycle time (analytic)",
    "estimate_from_logical_counts": "full physical estimate via Azure QRE (estimator)",
    "workload_counts": "published logical counts for a named algorithm (constant)",
}