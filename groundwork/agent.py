from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, TypedDict
from groundwork import verify as verify_mod
from langgraph.graph import END, START, StateGraph
from groundwork import domain
from groundwork.tools import lookup_constant
from groundwork.tools import dimension_of
from groundwork.models import (
    Answer,
    Assumption,
    Constant,
    Derivation,
    Derived,
    Quantity,
    Unknown,
    UserGiven,
    Check,
)

log = logging.getLogger(__name__)

MAX_COMPUTE_ATTEMPTS = 2


def _extend(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


class AgentState(TypedDict, total=False):
    question: str
    target_name: str # what we're solving for
    target_unit: str  # expected unit of the answer
    target_dimension: str  # expected dimension
    givens: dict[str, float]  # values the user actually supplied
    quantities: dict[str, Quantity]  # the DAG under construction
    steps: Annotated[list[str], _extend]
    assumptions: Annotated[list[Assumption], _extend]
    verification: dict[str, bool]     # check name -> passed
    verification_notes: Annotated[list[str], _extend]
    attempts: int
    answer: Answer
    workload: str | None
    leaf_kinds: dict[str, str]
    checks: list[Check]

LeafKind = Literal["user_input", "workload", "constant"]

# target -> (method, {input_name: how_to_ground_it})
RECIPES: dict[str, tuple[str, dict[str, "LeafKind"]]] = {
    "physical_qubits": ("estimator", {"logical_qubits": "workload", "t_count": "workload"}),
    "runtime": ("estimator", {"logical_qubits": "workload", "t_count": "workload"}),
    "code_distance": ("required_code_distance", {
        "physical_error_rate": "user_input",
        "target_logical_error_rate": "user_input",
    }),
    "logical_error_rate": ("logical_error_rate", {
        "physical_error_rate": "user_input",
        "code_distance": "user_input",
    }),
}
# nodes
import re

def parse(state: AgentState) -> dict[str, Any]:
    q = state["question"].lower()
    givens: dict[str, float] = {}

    # replace with structured LLM output later
    if m := re.search(r"physical error rate\s*(?:of\s*)?([\d.eE+-]+)", q):
        givens["physical_error_rate"] = float(m.group(1))
    if m := re.search(r"logical error rate\s*(?:of\s*)?([\d.eE+-]+)", q):
        givens["target_logical_error_rate"] = float(m.group(1))

    if "code distance" in q or "distance" in q:
        target, unit = "code_distance", ""
    elif "runtime" in q or "how long" in q:
        target, unit = "runtime", "s"
    else:
        target, unit = "physical_qubits", ""

    return {
        "target_name": target, "target_unit": unit,
        "target_dimension": dimension_of(unit) or "dimensionless",
        "givens": givens,
        "workload": "rsa2048_gidney2025" if "rsa" in q else None,
        "steps": [f"Parsed target: {target}, givens: {list(givens)}"],
        "attempts": 0,
    }


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
    return {
        "quantities": quantities,
        "leaf_kinds": dict(leaf_kinds),
        "steps": [f"Planned: {target} <- {method}({', '.join(leaf_kinds)})"],
        "assumptions": [
            Assumption(text="surface code QEC", impact="floquet code changes the footprint"),
            Assumption(text="gate-based ns qubits at 1e-3 error", impact="dominates physical qubit count"),
        ],
    }


def ground(state: AgentState) -> dict[str, Any]:
    target = state["target_name"]
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
                quantities[name] = Quantity(name=name, provenance=Unknown(
                    reason=f"no published workload counts for {name!r}"
                           + (f" in {workload!r}" if workload else " (no known algorithm named)")))
                notes.append(f"Could not ground {name} from workloads")

        elif kind == "constant":
            found = lookup_constant(name)
            quantities[name] = found
            notes.append(f"Grounded {name} from constants" if found.is_grounded
                         else f"No sourced constant for {name}")

    return {"quantities": quantities, "steps": notes}


def compute(state: AgentState) -> dict[str, Any]:
    target = state["target_name"]
    quantities = dict(state["quantities"])
    attempts = state.get("attempts", 0) + 1

    recipe = RECIPES.get(target)
    if recipe is None:
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
            results = domain.estimate_from_logical_counts(
                logical_qubits=int(args["logical_qubits"]),
                t_count=int(args["t_count"]),
                qubit_profile="qubit_gate_ns_e3",
                qec_scheme="surface_code",
                error_budget=0.01,
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

def verify(state: AgentState) -> dict[str, Any]:
    target = state["target_name"]
    quantities = state.get("quantities", {})
    result = quantities.get(target)

    checks: list[Check] = []
    if result is not None and result.is_grounded:
        checks.append(verify_mod.check_faithfulness(target, quantities, result))
        checks.append(verify_mod.check_dimensional(target, quantities, result))
        checks.append(verify_mod.check_magnitude(target, quantities, result))

    return {
        "checks": checks,
        "verification": {c.name: (c.status != "fail") for c in checks},   # keeps routing working
        "verification_notes": [f"{c.name}: {c.status} — {c.detail}" for c in checks],
    }


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

    if derivation.ungrounded:
        reasons = "; ".join(f"{q.name}: {q.provenance.reason}" for q in derivation.ungrounded)
        answer = Answer(status="abstained", question=state["question"], derivation=derivation,
                        checks=checks, abstain_reason=f"Cannot answer without: {reasons}")
    elif blocking_failed:
        why = "; ".join(f"{c.name}: {c.detail}" for c in blocking_failed)
        answer = Answer(status="abstained", question=state["question"], derivation=derivation,
                        checks=checks, abstain_reason=f"Verification failed — {why}")
    else:
        answer = Answer(status="answered", question=state["question"],
                        result=derivation.quantities[target], derivation=derivation,
                        checks=checks, confidence=0.8)
    return {"answer": answer}


# routing

def route_after_verify(state: AgentState) -> Literal["compute", "decide"]:
    failed = [n for n, ok in state.get("verification", {}).items() if not ok]
    if failed and state.get("attempts", 0) < MAX_COMPUTE_ATTEMPTS:
        log.info("verification failed (%s) — retrying compute", failed)
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


def run(question: str, callbacks: list[Any] | None = None) -> Answer:
    final = GRAPH.invoke(
        {"question": question},
        config={"callbacks": callbacks or []},
    )
    return final["answer"]