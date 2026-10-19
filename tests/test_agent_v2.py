# regression tests: each pins a failure the v2 evals found in v1

from __future__ import annotations

import json

import pytest

from groundwork import domain
from groundwork.agent import _extract_json, run
from groundwork.models import Derived, Quantity, UserGiven
from groundwork.verify import check_faithfulness, check_given_fidelity, numbers_in


@pytest.mark.parametrize("text,expected", [
    ("0.1%", [0.001]), ("0.0001%", [1e-6]), ("10^-3", [0.001]), ("10⁻¹⁵", [1e-15]),
    ("5×10^-4", [5e-4]), ("one in a thousand", [0.001]), ("rate 1e-4.", [1e-4]),
    ("rate -1e-3", [-0.001]), ("distance-25", [25.0]), ("1,000-qubit", [1000.0]),
])
def test_read_back_reader(text: str, expected: list[float]) -> None:
    assert numbers_in(text) == pytest.approx(expected)


def test_read_back_rejects_a_value_the_user_never_wrote() -> None:
    q = {"target_logical_error_rate": Quantity(name="target_logical_error_rate", value=1e-4, provenance=UserGiven())}
    assert check_given_fidelity("target logical error rate 0.0001%", q).blocks  # 0.0001% is 1e-6


def test_faithfulness_is_relative_below_one() -> None:
    # v1: |3e-14 - 3e-15| <= 1e-9 * max(1, 3e-14) passed. A 10x error is not a match.
    stated = Quantity(name="logical_error_rate", value=3e-14,
                      provenance=Derived(method="analytic:surface_code_scaling[surface_code]",
                                         inputs=["physical_error_rate", "code_distance"]))
    qs = {"physical_error_rate": Quantity(name="physical_error_rate", value=1e-3, provenance=UserGiven()),
          "code_distance": Quantity(name="code_distance", value=25, provenance=UserGiven()),
          "logical_error_rate": stated}
    assert check_faithfulness("logical_error_rate", qs, stated).status == "fail"


def test_extract_json_tolerates_prose_and_think_tags() -> None:
    body = {"target": "code_distance"}
    assert json.loads(_extract_json("Here you go:\n" + json.dumps(body))) == body
    assert json.loads(_extract_json("<think>{not this}</think>" + json.dumps(body))) == body


def test_toffolis_go_to_the_ccz_slot() -> None:
    prog = domain._logical_counts_program(1409, 6_500_000_000)
    assert "CczCount(6500000000)" in prog and "TCount(0)" in prog


def test_gidney_ekera_2019_count_matches_its_abstract() -> None:
    n = 2048
    assert domain.WORKLOADS["rsa2048_gidney2019"].toffoli_count == int(0.3 * n**3 + 0.0005 * n**3 * 11)


def test_workloads_match_key_size_exactly() -> None:
    assert domain.find_workload("rsa", 2048) == "rsa2048_gidney2025"
    assert domain.find_workload("rsa", 4096) is None


@pytest.mark.parametrize("question", [
    "How many physical qubits to factor a 4096-bit RSA key?",
    "What code distance do I need at physical error rate 1e-3 and target logical error rate 0.0001%?",
    "How many physical qubits to factor RSA-2048 on Majorana qubits with a physical error rate of 1e-6?",
])
def test_regex_reader_abstains_instead_of_guessing(question: str) -> None:
    assert run(question, use_llm=False).status == "abstained"


def test_sentence_final_number_does_not_crash() -> None:
    a = run("I want a target logical error rate of 1e-12 with qubits at physical error rate 1e-4. What code distance?", use_llm=False)
    assert a.status == "answered" and a.result is not None and a.result.value == 11
