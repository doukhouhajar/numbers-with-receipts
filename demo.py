from __future__ import annotations

import argparse
import json
import sys

from groundwork.agent import run
from groundwork.config import get_langfuse_handler, setup_logging
from groundwork.models import Answer, Constant, Derived, Quantity, UserGiven

DEFAULT_QUESTION = ("What is the runtime to factor RSA-2048 at a physical error rate of 1e-3?")
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


#def rule(title: str = "") -> str:
#   return f"{DIM}{'─' * 4} {title} {'─' * max(0, 66 - len(title))}{RESET}"


def receipt(q: Quantity) -> str:
    p = q.provenance
    if isinstance(p, UserGiven):
        return f"{GREEN}given by user{RESET}"
    if isinstance(p, Constant):
        return f"{GREEN}constant{RESET} {DIM}← {p.source}{RESET}"
    if isinstance(p, Derived):
        code = f" {DIM}[{p.code_ref}]{RESET}" if p.code_ref else ""
        return f"{CYAN}computed{RESET} via {p.method} {DIM}← {', '.join(p.inputs)}{RESET}{code}"
    return f"{RED}UNKNOWN{RESET} {DIM}({p.reason}){RESET}"


def fmt_value(q: Quantity) -> str:
    if q.value is None:
        return f"{RED}—{RESET}"
    v = f"{q.value:,.4g}"
    if q.uncertainty:
        v += f" ± {q.uncertainty:.2g}"
    return f"{BOLD}{v}{RESET}{(' ' + q.unit) if q.unit else ''}"


def render(answer: Answer) -> str:
    out: list[str] = ["", "QUESTION", answer.question, ""]

    d = answer.derivation
    out.append("DERIVATION")
    # dependencies before dependents
    ordered = sorted(
        d.quantities.values(),
        key=lambda q: (isinstance(q.provenance, Derived), q.name),
    )
    for q in ordered:
        marker = "→" if q.name == d.target else " "
        out.append(f" {marker} {q.name:<24} {fmt_value(q):<28} {receipt(q)}")

    if d.steps:
        out += ["", "STEPS"] + [f"   {i}. {s}" for i, s in enumerate(d.steps, 1)]

    if d.assumptions:
        out += ["", "ASSUMPTIONS"]
        for a in d.assumptions:
            out.append(f"   • {a.text}{f' {DIM}({a.impact}){RESET}' if a.impact else ''}")

    out += ["", "VERDICT"]
    if answer.status == "answered":
        assert answer.result is not None
        conf = f" {DIM}(confidence {answer.confidence:.0%}){RESET}" if answer.confidence else ""
        out.append(f" {GREEN}{BOLD}ANSWERED{RESET}  {answer.result.name} = {fmt_value(answer.result)}{conf}")
    else:
        out.append(f" {YELLOW}{BOLD}ABSTAINED{RESET}  {answer.abstain_reason}")
    out.append("")
    return "\n".join(out)


# entrypoint

def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Groundwork agent")
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("--json", action="store_true", help="emit the raw Answer as JSON")
    ap.add_argument("--no-trace", action="store_true", help="skip Langfuse tracing")
    args = ap.parse_args()

    setup_logging()

    callbacks = []
    if not args.no_trace:
        handler = get_langfuse_handler()
        if handler is not None:
            callbacks.append(handler)
        else:
            print(f"{DIM}(Langfuse not configured, running untraced){RESET}", file=sys.stderr)

    answer = run(args.question, callbacks=callbacks)

    if args.json:
        print(json.dumps(answer.model_dump(mode="json"), indent=2))
    else:
        print(render(answer))

    # exit 0 = answered, 2 = abstained
    return 0 if answer.status == "answered" else 2


if __name__ == "__main__":
    raise SystemExit(main())