from __future__ import annotations
import asyncio
import json
from collections.abc import Callable
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    chain,
    generate,
    prompt_template,
    solver,
    system_message,
)

from groundwork.agent import run

@solver
def groundwork_agent(use_llm: bool = True, reader: Callable[[TaskState], Callable[[str], str]] | None = None) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        loop = asyncio.get_running_loop()
        model = get_model()

        def llm(prompt: str) -> str:
            future = asyncio.run_coroutine_threadsafe(model.generate(prompt), loop)
            return future.result(timeout=300).completion

        fn = reader(state) if reader is not None else llm
        try:
            answer = await asyncio.to_thread(run, state.input_text, llm=fn, use_llm=use_llm)
            completion = answer.model_dump_json()
        except Exception as exc:  # a crash is a finding, scored as "error"
            completion = json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        state.output = ModelOutput.from_content("groundwork", completion)
        return state

    return solve


# model alone 
CONVENTION = (
    "You answer quantitative questions about fault-tolerant quantum computing.\n"
    "For surface-code questions use this convention: the logical error rate is "
    "P_L = 0.03 * (p / 0.01) ** ((d + 1) / 2), where p is the physical error rate and d is an "
    "odd code distance; the required code distance is the smallest odd d with P_L <= the target. "
    "Give durations in seconds and counts as plain numbers."
)
CONTRACT = (
    "\n\nEnd your reply with one final line of the form `ANSWER: <number>`. "
    "If you do not give a number, end with `ANSWER: NONE`."
)
ABSTAIN_RULE = (
    "\nIf the question is missing information needed for a definite number, if its inputs are "
    "outside the range where the formula is valid, or if answering would need figures you "
    "cannot source, do not estimate: say what is missing and end with `ANSWER: ABSTAIN`."
)


@solver
def bare(abstain_instruction: bool = False) -> Solver:
    system = CONVENTION + (ABSTAIN_RULE if abstain_instruction else "")
    return chain(
        system_message(system),
        prompt_template("{prompt}" + CONTRACT),
        generate(),
    )


# test the eval 
@solver
def dummy(kind: str = "oracle") -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        if kind == "oracle" and md["expected"] == "answer":
            value = md["oracle_value"]
            if value is None:  # range label: answer the middle of the band
                lo, hi = md["range"]
                value = (lo * hi) ** 0.5
            text = f"ANSWER: {value!r}"
        else:
            text = "ANSWER: ABSTAIN"
        state.output = ModelOutput.from_content("dummy", text)
        return state

    return solve
