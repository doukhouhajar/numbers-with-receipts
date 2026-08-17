import logging

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from groundwork.agent import run
from groundwork.models import Answer


class EstimateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

class EstimateResponse(BaseModel):
    answered: bool
    value: float | None
    answer: Answer   

    @classmethod
    def from_answer(cls, a: Answer) -> "EstimateResponse":
        return cls(
            answered=a.status == "answered",
            value=a.result.value if a.result else None,
            answer=a,
        )
    

log = logging.getLogger(__name__)
app = FastAPI(title="Groundwork", description="Quantum resource estimates with receipts ;)")

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/estimate", response_model=EstimateResponse)
async def estimate(req: EstimateRequest) -> EstimateResponse:
    log.info("estimate: %s", req.question)
    answer = await run_in_threadpool(run, req.question)
    return EstimateResponse.from_answer(answer)

@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Groundwork", "docs": "/docs", "health": "/health"}