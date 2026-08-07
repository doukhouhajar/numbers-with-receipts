from __future__ import annotations

import asyncio
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    accuracy,
    mean,
    metric,
    scorer,
    stderr,
    value_to_float,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from groundwork import domain
from groundwork.agent import RECIPES, run
from groundwork.models import Answer, Derived

DATA = Path(__file__).parent / "datasets"


def _load(name: str) -> MemoryDataset:
    samples = []
    for line in (DATA / name).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        samples.append(
            Sample(input=rec["input"], target=str(rec.get("target", "")), metadata=rec.get("metadata", {}))
        )
    return MemoryDataset(samples)


# run the whole agent
@solver
def groundwork_agent() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.input_text
        try:
            answer = run(question)
            completion = answer.model_dump_json()
        except Exception as exc:  # a crash is itself a finding 
            completion = json.dumps({"status": "error", "error": str(exc)})
        state.output = ModelOutput.from_content("groundwork", completion)
        return state

    return solve


def _answer(state: TaskState) -> Answer | None:
    try:
        return Answer.model_validate_json(state.output.completion)
    except Exception:
        return None

# accuracy() counts NOANSWER as 0. We want NOANSWER to mean "not applicable to this sample" and be excluded from the denominator
@metric
def applicable_accuracy() -> Metric:
    to_float = value_to_float()

    def compute(scores: list[SampleScore]) -> Value:
        applicable = [
            s.score for s in scores
            if (s.score.metadata or {}).get("applicable", True)
        ]
        if not applicable:
            return 0.0
        return sum(to_float(s.value) for s in applicable) / len(applicable)

    return compute

@scorer(metrics=[accuracy(), stderr()])
def grounded_accuracy() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        a = _answer(state)
        if a is None:
            return Score(value=INCORRECT, explanation="agent errored")
        if a.status != "answered" or a.result is None:
            return Score(value=INCORRECT, explanation=f"abstained: {a.abstain_reason}")
        val = a.result.value
        meta = state.metadata
        if "target_value" in meta:
            tgt, tol = float(meta["target_value"]), float(meta.get("rel_tol", 0.1))
            ok = abs(val - tgt) <= tol * abs(tgt) or abs(val - tgt) < 1
            return Score(value=CORRECT if ok else INCORRECT, answer=str(val),
                         explanation=f"got {val}, expected {tgt} (±{tol:.0%})")
        if "range" in meta:
            lo, hi = meta["range"]
            ok = lo <= val <= hi
            return Score(value=CORRECT if ok else INCORRECT, answer=str(val),
                         explanation=f"got {val}, plausible [{lo}, {hi}]")
        return Score(value=NOANSWER, explanation="no target specified")

    return score


@scorer(metrics=[accuracy(), stderr()])
def appropriate_abstention() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        a = _answer(state)
        if a is None:
            return Score(value=INCORRECT, explanation="agent errored (crash, not a clean abstention)")
        return Score(value=CORRECT if a.status == "abstained" else INCORRECT,
                     explanation=a.abstain_reason or f"status={a.status}")

    return score


@scorer(metrics=[mean()])
def hallucination_rate() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        a = _answer(state)
        hallucinated = 1.0 if (a is not None and a.status == "answered") else 0.0
        return Score(value=hallucinated,
                     explanation="answered when it should abstain" if hallucinated else "no fabricated number")

    return score


@scorer(metrics=[applicable_accuracy()])
def faithfulness() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        def na(detail: str) -> Score:
            return Score(value=NOANSWER, explanation=detail, metadata={"applicable": False})

        a = _answer(state)
        if a is None:
            return na("agent produced no parseable answer")
        check = next((c for c in a.checks if c.name == "faithfulness"), None)
        if check is None:
            return na("NO faithfulness check on answer" if a.status == "answered"
                      else "abstained, nothing to check")
        if check.status == "na":
            return na(check.detail)
        return Score(
            value=CORRECT if check.status == "pass" else INCORRECT,
            explanation=check.detail,
            metadata={"applicable": True},
        )
    return score


@task
def answerable() -> Task:
    return Task(dataset=_load("answerable.jsonl"), solver=groundwork_agent(),
                scorer=[grounded_accuracy(), faithfulness()])


@task
def underspecified() -> Task:
    return Task(dataset=_load("underspecified.jsonl"), solver=groundwork_agent(),
                scorer=[appropriate_abstention(), hallucination_rate()])


@task
def unit_traps() -> Task:
    return Task(dataset=_load("unit_traps.jsonl"), solver=groundwork_agent(),
                scorer=[appropriate_abstention(), hallucination_rate()])


def _selftest_metric() -> None:
    from inspect_ai.scorer import SampleScore, Score
    mk = lambda v, ok: SampleScore(score=Score(value=v, metadata={"applicable": ok}))
    m = applicable_accuracy()
    assert m([mk(CORRECT, True)] * 4 + [mk(NOANSWER, False)] * 2) == 1.0
    assert m([mk(CORRECT, True), mk(INCORRECT, True)]) == 0.5


if __name__ == "__main__":
    _selftest_metric()
    inspect_eval([answerable(), underspecified(), unit_traps()], model="mockllm/model")