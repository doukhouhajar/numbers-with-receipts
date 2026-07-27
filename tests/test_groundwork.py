from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundwork.models import (
    Answer,
    Assumption,
    Constant,
    Derivation,
    Derived,
    Quantity,
    Unknown,
    UserGiven,
)


def given(name: str, value: float, unit: str = "", dimension: str = "") -> Quantity:
    return Quantity(
        name=name, value=value, unit=unit, dimension=dimension, provenance=UserGiven()
    )


def derived(name: str, value: float, inputs: list[str], unit: str = "") -> Quantity:
    return Quantity(
        name=name,
        value=value,
        unit=unit,
        provenance=Derived(method="sandbox:numpy", inputs=inputs, code_ref="x = a * b"),
    )



def test_unknown_provenance_cannot_carry_a_value() -> None:
    with pytest.raises(ValidationError, match="Unknown provenance"):
        Quantity(
            name="physical_error_rate",
            value=1e-3,
            provenance=Unknown(reason="user never specified a hardware profile"),
        )


def test_grounded_provenance_requires_a_value() -> None:
    with pytest.raises(ValidationError):
        Quantity(name="code_distance", value=None, provenance=UserGiven())


def test_constant_requires_a_source() -> None:
    with pytest.raises(ValidationError):
        Constant(source="")
    with pytest.raises(ValidationError):
        Constant()  # type: ignore[call-arg]


def test_derived_requires_method_and_inputs() -> None:
    with pytest.raises(ValidationError):
        Derived(method="qsharp.estimate", inputs=[])
    with pytest.raises(ValidationError):
        Derived(method="", inputs=["a"])


def test_unknown_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        Unknown(reason="")


def test_valid_quantities_construct() -> None:
    q = Quantity(
        name="p_err",
        value=1e-3,
        unit="",
        dimension="dimensionless",
        provenance=Constant(source="user hardware profile: qubit_gate_ns_e3"),
    )
    assert q.is_grounded
    assert Quantity(name="x", provenance=Unknown(reason="missing")).is_grounded is False


def test_provenance_discriminator_round_trips() -> None:
    q = derived("runtime", 3600.0, ["p_err"], unit="s")
    restored = Quantity.model_validate_json(q.model_dump_json())
    assert isinstance(restored.provenance, Derived)
    assert restored.provenance.method == "sandbox:numpy"



def test_derivation_rejects_dangling_inputs() -> None:
    with pytest.raises(ValidationError, match="unknown quantities"):
        Derivation(
            target="runtime",
            quantities={"runtime": derived("runtime", 10.0, ["nonexistent"])},
        )


def test_derivation_rejects_cycles() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Derivation(
            target="a",
            quantities={
                "a": derived("a", 1.0, ["b"]),
                "b": derived("b", 2.0, ["a"]),
            },
        )


def test_derivation_rejects_key_name_mismatch() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        Derivation(target="a", quantities={"wrong_key": given("a", 1.0)})


def test_valid_derivation_and_ungrounded_property() -> None:
    d = Derivation(
        target="runtime",
        quantities={
            "p_err": given("p_err", 1e-3),
            "n_qubits": Quantity(name="n_qubits", provenance=Unknown(reason="not given")),
            "runtime": derived("runtime", 3600.0, ["p_err"], unit="s"),
        },
        steps=["runtime = f(p_err)"],
        assumptions=[Assumption(text="surface code", impact="changes overhead")],
    )
    assert [q.name for q in d.ungrounded] == ["n_qubits"]


def _answerable() -> Derivation:
    return Derivation(
        target="runtime",
        quantities={
            "p_err": given("p_err", 1e-3),
            "runtime": derived("runtime", 3600.0, ["p_err"], unit="s"),
        },
    )


def test_cannot_answer_while_something_is_ungrounded() -> None:
    d = Derivation(
        target="runtime",
        quantities={
            "p_err": Quantity(name="p_err", provenance=Unknown(reason="not given")),
            "runtime": derived("runtime", 3600.0, ["p_err"], unit="s"),
        },
    )
    with pytest.raises(ValidationError, match="ungrounded"):
        Answer(status="answered", result=d.quantities["runtime"], derivation=d)


def test_cannot_answer_without_a_result() -> None:
    with pytest.raises(ValidationError, match="no result"):
        Answer(status="answered", derivation=_answerable())


def test_cannot_abstain_without_a_reason() -> None:
    with pytest.raises(ValidationError, match="without a reason"):
        Answer(status="abstained", derivation=_answerable())


def test_cannot_abstain_while_attaching_a_result() -> None:
    d = _answerable()
    with pytest.raises(ValidationError, match="result was attached"):
        Answer(
            status="abstained",
            result=d.quantities["runtime"],
            derivation=d,
            abstain_reason="missing hardware profile",
        )


def test_valid_answer_and_valid_abstention() -> None:
    d = _answerable()
    ok = Answer(
        status="answered",
        question="runtime to factor RSA-2048?",
        result=d.quantities["runtime"],
        derivation=d,
        confidence=0.8,
    )
    assert ok.result is not None and ok.result.is_grounded

    no = Answer(
        status="abstained",
        derivation=Derivation(target="runtime"),
        abstain_reason="no physical error rate supplied",
    )
    assert no.result is None


def test_confidence_is_bounded() -> None:
    d = _answerable()
    with pytest.raises(ValidationError):
        Answer(status="answered", result=d.quantities["runtime"], derivation=d, confidence=1.5)