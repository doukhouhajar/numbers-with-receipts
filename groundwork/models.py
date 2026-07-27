from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class UserGiven(BaseModel):
    kind: Literal["given"] = "given"


class Constant(BaseModel):
    kind: Literal["constant"] = "constant"
    source: str = Field(min_length=1) 


class Derived(BaseModel):
    kind: Literal["derived"] = "derived"
    method: str = Field(min_length=1)  # "qsharp.estimate", "sympy.solve", "sandbox:numpy"
    inputs: list[str] = Field(min_length=1)  # names of quantities this depends on
    code_ref: str | None = None  # the code that produced it


class Unknown(BaseModel):
    kind: Literal["unknown"] = "unknown"
    reason: str = Field(min_length=1)


Provenance = Annotated[
    UserGiven | Constant | Derived | Unknown,
    Field(discriminator="kind"),
]


class Quantity(BaseModel):
    name: str = Field(min_length=1)
    value: float | None = None  # None iff provenance is Unknown
    unit: str = ""  # "" := dimensionless
    dimension: str = ""  # expected dimension
    uncertainty: float | None = Field(default=None, ge=0)
    provenance: Provenance

    @model_validator(mode="after")
    def _value_matches_provenance(self) -> Quantity:
        is_unknown = self.provenance.kind == "unknown"
        if is_unknown and self.value is not None:
            raise ValueError(
                f"{self.name!r}: has a value but Unknown provenance, "
                "this is exactly the hallucination we refuse to allow."
            )
        if not is_unknown and self.value is None:
            raise ValueError(f"{self.name!r}: grounded provenance requires a value.")
        return self

    @property
    def is_grounded(self) -> bool:
        return self.provenance.kind != "unknown"


class Assumption(BaseModel):
    text: str = Field(min_length=1)
    impact: str = ""  # why it matters


class Derivation(BaseModel):
    target: str = Field(min_length=1)  # name of the quantity being solved for
    quantities: dict[str, Quantity] = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)

    @model_validator(mode="after")
    def _dag_is_closed_and_acyclic(self) -> Derivation:
        for name, q in self.quantities.items():
            if name != q.name:
                raise ValueError(f"key {name!r} does not match Quantity.name {q.name!r}")
            if isinstance(q.provenance, Derived):
                missing = [i for i in q.provenance.inputs if i not in self.quantities]
                if missing:
                    raise ValueError(f"{name!r} depends on unknown quantities: {missing}")

        # depth-first cycle detection over Derived edges
        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(self.quantities, WHITE)

        def visit(n: str, trail: list[str]) -> None:
            if colour[n] == GREY:
                raise ValueError(f"cycle in derivation: {' -> '.join(trail + [n])}")
            if colour[n] == BLACK:
                return
            colour[n] = GREY
            prov = self.quantities[n].provenance
            if isinstance(prov, Derived):
                for dep in prov.inputs:
                    visit(dep, trail + [n])
            colour[n] = BLACK

        for n in self.quantities:
            visit(n, [])
        return self

    @property
    def ungrounded(self) -> list[Quantity]:
        return [q for q in self.quantities.values() if not q.is_grounded]


class Answer(BaseModel):
    status: Literal["answered", "abstained"]
    question: str = ""
    result: Quantity | None = None
    derivation: Derivation
    abstain_reason: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _status_is_coherent(self) -> Answer:
        if self.status == "answered":
            if self.result is None:
                raise ValueError("answered but no result")
            if not self.result.is_grounded:
                raise ValueError("answered with an ungrounded result")
            if self.derivation.ungrounded:
                names = [q.name for q in self.derivation.ungrounded]
                raise ValueError(f"answered while these remain ungrounded: {names}")
        else:
            if not self.abstain_reason:
                raise ValueError("abstained without a reason")
            if self.result is not None:
                raise ValueError("abstained but a result was attached")
        return self
    


"""
Invariants enforced here:
  1. A Quantity with a real value cannot have Unknown provenance.
  2. A Constant cannot be constructed without a source.
  3. A Derived quantity must name its method and its input quantities.
  4. A Derivation's DAG must be closed (no dangling inputs) and acyclic.
  5. An Answer is either answered-with-a-result or abstained-with-a-reason.
"""