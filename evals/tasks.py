from __future__ import annotations
import json
import sys
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

sys.path.insert(0, str(Path(__file__).parent))  

from scorers import (  # noqa: E402
    assumption_fidelity,
    checks_fired,
    outcome,
    parse_path,
    receipt_fidelity,
)
from solvers import bare, dummy, groundwork_agent  # noqa: E402

DATA = Path(__file__).parent / "datasets" / "receipts_v2.jsonl"
SYSTEMS = ("groundwork", "regex", "bare", "bare_abstain", "dummy_oracle", "dummy_abstain")


def load(families: str = "all", exclude: tuple[str, ...] = ()) -> MemoryDataset:
    wanted = None if families == "all" else set(families.split(","))
    samples = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        fam = row["metadata"]["family"]
        if (wanted is None or fam in wanted) and fam not in exclude:
            samples.append(Sample(id=row["id"], input=row["input"], target=row["target"],
                                  metadata=row["metadata"]))
    return MemoryDataset(samples, name="receipts_v2")


@task
def receipts(system: str = "groundwork", families: str = "all") -> Task:
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}")
    solver = {
        "groundwork": groundwork_agent(use_llm=True),
        "regex": groundwork_agent(use_llm=False),
        "bare": bare(abstain_instruction=False),
        "bare_abstain": bare(abstain_instruction=True),
        "dummy_oracle": dummy("oracle"),
        "dummy_abstain": dummy("abstain"),
    }[system]
    # F7 asks whether the derivation ran on the machine the user named
    exclude = ("F7",) if system.startswith("bare") else ()
    return Task(
        dataset=load(families, exclude),
        solver=solver,
        scorer=[outcome(), receipt_fidelity(), parse_path(), assumption_fidelity()],
        config=GenerateConfig(temperature=0.0, max_tokens=1024),
        metadata={"system": system},
    )


# evaluate the evaluator 
MUTATION_FAMILIES = ("F1", "F2", "F6")


def truth_reply(md: dict, mutant: str) -> str:
    t = dict(md["truth"])
    if mutant == "M7_reader_swaps_inputs" and t.get("physical_error_rate") is not None:
        other = "target_logical_error_rate" if t.get("target_logical_error_rate") is not None else "code_distance"
        t["physical_error_rate"], t[other] = t[other], t["physical_error_rate"]
    if mutant == "M8_reader_percent_blind" and t.get("physical_error_rate") is not None:
        t["physical_error_rate"] = t["physical_error_rate"] * 100  # "0.1%" read as 0.1
    return json.dumps(t)


@task
def mutation(mutant: str = "none") -> Task:
    rows = [s for s in load(",".join(MUTATION_FAMILIES)) if s.metadata.get("truth")
            and s.metadata["expected"] == "answer"]

    def reader(state):  # a scripted reader
        reply = truth_reply(state.metadata, mutant)
        return lambda prompt: reply

    return Task(
        dataset=MemoryDataset(rows, name=f"mutation-{mutant}"),
        solver=groundwork_agent(use_llm=True, reader=reader),
        scorer=[checks_fired(), outcome(), receipt_fidelity()],
        metadata={"mutant": mutant},
    )
