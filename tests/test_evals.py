# eval of the eval
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).parent.parent / "evals"
sys.path.insert(0, str(EVALS))

import oracle  # noqa: E402
from scorers import Reading, classify, parse_number, read_completion, receipt_verdict  # noqa: E402


def test_oracle_matches_hand_computed_values() -> None:
    oracle.selftest()


def test_oracle_agrees_with_every_label_in_the_dataset() -> None:
    for line in (EVALS / "datasets" / "receipts_v2.jsonl").read_text().splitlines():
        md = json.loads(line)["metadata"]
        g = md["expected_givens"]
        if md["target_name"] == "code_distance" and md["expected"] == "answer":
            assert md["oracle_value"] == oracle.code_distance(g["physical_error_rate"], g["target_logical_error_rate"])


@pytest.mark.parametrize("text,expected", [
    ("21", 21), ("d = 27", 27), ("5,321,372", 5321372), ("3.0e-13", 3e-13), ("3 × 10^-13", 3e-13),
    ("3x10^(-13)", 3e-13), ("10^-9", 1e-9), ("10⁻⁹", 1e-9), ("2.5 × 10⁻⁷", 2.5e-7),
    ("20 million", 2e7), ("about 1 billion", 1e9), ("3e−9", 3e-9),
])
def test_parse_number(text: str, expected: float) -> None:
    assert parse_number(text) == pytest.approx(expected)


def test_parse_number_converts_durations_only_when_asked() -> None:
    assert parse_number("8 hours", as_seconds=True) == pytest.approx(28800)
    assert parse_number("8 hours") == pytest.approx(8)


def test_read_completion_uses_the_last_answer_line() -> None:
    r = read_completion("ANSWER: 7 is tempting, but...\nANSWER: 9")
    assert (r.status, r.value) == ("answered", 9)
    assert read_completion("I won't guess.\nANSWER: ABSTAIN").status == "abstained"
    assert read_completion("no contract here").status == "error"


ANSWER_ROW = {"expected": "answer", "oracle_value": 27, "rel_tol": 0.0}
ABSTAIN_ROW = {"expected": "abstain"}


@pytest.mark.parametrize("reading,md,cell", [
    (Reading("bare", "answered", 27.0, None), ANSWER_ROW, "correct"),
    (Reading("bare", "answered", 25.0, None), ANSWER_ROW, "wrong"),  # d=25 is not d=27: exact match
    (Reading("bare", "abstained", None, None), ANSWER_ROW, "over_abstain"),
    (Reading("bare", "abstained", None, None), ABSTAIN_ROW, "proper_abstain"),
    (Reading("bare", "answered", 13.0, None), ABSTAIN_ROW, "fabricated"),
    (Reading("bare", "error", None, None), ANSWER_ROW, "error"),
])
def test_every_outcome_cell_is_reachable(reading: Reading, md: dict, cell: str) -> None:
    assert classify(reading, md) == cell


def test_receipts() -> None:
    said = {"physical_error_rate": 1e-3, "target_logical_error_rate": 1e-6}
    assert receipt_verdict(said, said)[0] == "faithful"
    assert receipt_verdict({**said, "target_logical_error_rate": 1e-4}, said)[0] == "forged"  # 0.0001% read as 1e-4
    assert receipt_verdict({"physical_error_rate": 1e-3}, said)[0] == "missing"
    assert receipt_verdict({"code_distance": 15}, {})[0] == "forged"  # a number the user never gave


def _rates(system: str) -> dict[str, float]:
    from inspect_ai import eval as inspect_eval
    from tasks import receipts

    log = inspect_eval(receipts(system=system), model="mockllm/model", display="none", log_dir=None)[0]
    out = log.results.scores[0].metrics
    return {k: v.value for k, v in out.items()}


def test_dummy_oracle_scores_perfectly() -> None:
    r = _rates("dummy_oracle")
    assert r["correct"] + r["proper_abstain"] == pytest.approx(1.0)
    assert r["fabricated"] == r["wrong"] == r["over_abstain"] == r["error"] == 0


def test_dummy_abstain_never_answers() -> None:
    r = _rates("dummy_abstain")
    assert r["correct"] == r["wrong"] == r["fabricated"] == 0
    assert r["over_abstain"] > 0 and r["proper_abstain"] > 0
