# numbers with receipts

*An agent that refuses to make up numbers. Every value it reports carries a receipt; when it can't ground an answer, it abstains instead of inventing one.*

## Thesis

Ask a language model a quantitative question and it will always give you a number. The failure is that **there's no way to tell a computed number from a hallucinated one.** 
This project makes that distinction clearer. Every number in the system is a `Quantity` that must carry its **provenance**:

| Provenance | Meaning | Requirement |
|------------|---------|-------------|
| `UserGiven` | the user supplied it | — |
| `Constant`  | a sourced value | must name its **source** (a citation) |
| `Derived`   | computed | must name its **method** and its **input** quantities |
| `Unknown`   | not grounded | must name a **reason** and **cannot carry a value** |


An `Answer` is likewise **answered-with-a-grounded-result** or **abstained-with-a-reason**. A hallucinated answer isn't caught at runtime; it **fails to typecheck**. See [`groundwork/models.py`](groundwork/models.py) for the five invariants.

## How it works

A [LangGraph](https://github.com/langchain-ai/langgraph) agent runs six typed nodes. The number is built up a leaf at a time, each leaf grounded before anything is computed, and the result is re-checked.

```mermaid
flowchart LR
    parse["parse<br/><i>NL → structured request</i>"] --> plan["plan<br/><i>build the DAG</i>"]
    plan --> ground["ground<br/><i>resolve every leaf</i>"]
    ground --> compute["compute<br/><i>run the estimator</i>"]
    compute --> verify["verify<br/><i>independent checks</i>"]
    verify -->|checks fail| compute
    verify -->|ok / exhausted| decide["decide<br/><i>answer or abstain</i>"]
```

- **parse** — an LLM turns the question into a structured request. No LLM available? It falls back to a regex parser.
- **plan** — picks a recipe and lays out the computation DAG, tagging each leaf as *user input*, *published workload* or *sourced constant*.
- **ground** — resolves the leaves. A missing required input becomes `Unknown`.
- **compute** — runs the domain method (analytic formula or resource estimator) and tags the result `Derived`.
- **verify** — independent checks (below). Advisory checks annotate; blocking checks can force a recompute or an abstention.
- **decide** — assembles the `Derivation` and emits an `answered` or `abstained` `Answer`.

## Domain: quantum resource estimation

The spine is domain-agnostic; the wired-up domain is **fault-tolerant quantum resource estimation**; "how many physical qubits / how long to run algorithm X on hardware Y?" It's a good stress test: the numbers span many orders of magnitude, the literature disagrees and a plausible-looking wrong answer is easy to produce.

Grounding sources, each receipted:

- **Analytic QEC formulas** — surface-code logical error rate `P_L ≈ a·(p/p_th)^((d+1)/2)`, required code distance (inverted and forward-verified), the `2d²` footprint, cycle time — each `code_ref`'d to Fowler et al. (2012) / the Azure QRE methodology.
- **Azure Quantum Resource Estimator** (`qsharp.estimate`) — full physical estimates from logical counts, run locally.
- **Published workloads** — logical qubit / Toffoli counts for named algorithms (e.g. RSA-2048 from Gidney 2025 and Gidney–Ekerå 2019), stored as `Constant`s with their citations.
- **Cross-method check** — [Qualtran](https://github.com/quantumlib/Qualtran) as a second, independent estimator; the two must agree within tolerance.


## Verification layer

Grounding says a number *has* a source. Verification asks whether it's *right*. Each check is a `Check` with a severity ([`groundwork/verify.py`](groundwork/verify.py)):

| Check | Asks | Severity |
|-------|------|----------|
| **faithfulness** | re-execute the derivation — does it reproduce the stated value? | blocking |
| **dimensional** | do the units match the expected dimension? (via [Pint](https://pint.readthedocs.io)) | advisory |
| **magnitude** | is the value inside a physically plausible band? | advisory |
| **cross-method** | does an independent estimator (Qualtran) agree within tolerance? | advisory |

A failed **blocking** check triggers a recompute and, if unresolved, an abstention.

## Quickstart 

```bash
git clone https://github.com/doukhouhajar/numbers-with-receipts.git
cd numbers-with-receipts
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     
cp .env.example .env           
python demo.py                                  
python demo.py "How many physical qubits to factor RSA-2048?"
python demo.py --json "..."                     
```

**Config is all optional.** With an empty `.env` the agent still runs: the parser falls back to regex and tracing is disabled. To get the most out of it:

- **LLM parsing** — set `LLM_PROVIDER=ollama` with a local model (`ollama pull qwen3:8b`), *or* `LLM_PROVIDER=openai-compatible` pointed at any OpenAI-style endpoint (vLLM, a hosted API, …). See `.env.example`.
- **Sandboxed execution** — an `E2B_API_KEY` ([e2b.dev](https://e2b.dev)) for the code-execution tool.
- **Tracing** — Langfuse keys to get one span per node.

> **Hardware note.** The Azure estimator and a local LLM are the heavy parts. On a small laptop, run with the regex parser (no LLM) or point `LLM_PROVIDER` at a hosted endpoint, neither needs a GPU. 

## HTTP API

```bash
uvicorn groundwork.api:app --reload
```
- `POST /estimate` — `{"question": "..."}` → a structured `Answer` (result + full derivation + checks)
- `GET  /health` — liveness
- `GET  /docs` — OpenAPI UI

## Evals

An [Inspect](https://inspect.aisi.org.uk/) harness ([`evals/eval.py`](evals/eval.py)) scores the agent on curated sets — `answerable`, `underspecified`, `unit_traps`, `magnitude_traps` — measuring grounded accuracy, appropriate-abstention rate, and hallucination rate.

```bash
inspect eval evals/eval.py
```

Note that hallucination is bounded by construction, not just empirically: an `answered` `Answer` with an ungrounded result cannot be built. The eval measures the harder axis; whether the agent abstains too much or too little.

## Project layout

```
groundwork/
  models.py    # the typed spine: Quantity, Provenance, Derivation, Answer + invariants
  agent.py     # LangGraph graph: parse → plan → ground → compute → verify → decide
  domain.py    # quantum RE: analytic formulas, Azure QRE wrapper, published workloads
  verify.py    # faithfulness / dimensional / magnitude / cross-method checks
  tools.py     # Pint unit registry, sourced-constant lookup, E2B sandbox
  config.py    # pydantic-settings; LLM + tracing wiring
  api.py       # FastAPI service
demo.py        # pretty-printed end-to-end run
evals/         # Inspect tasks + datasets
tests/         # provenance-invariant tests
```

## License

MIT
