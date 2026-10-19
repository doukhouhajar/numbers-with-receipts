from __future__ import annotations

import hashlib
import json
from pathlib import Path

import oracle

OUT = Path(__file__).parent / "datasets" / "receipts_v2.jsonl"

P = {"1e-4": 1e-4, "5e-4": 5e-4, "1e-3": 1e-3, "2e-3": 2e-3, "5e-3": 5e-3}
T = {"1e-6": 1e-6, "1e-9": 1e-9, "1e-12": 1e-12, "1e-15": 1e-15}

QUBITS_RANGE = [1e5, 1e8]   # physical qubits for RSA-2048 across published estimates
RUNTIME_RANGE = [1e2, 1e7]  # seconds

rows: list[dict] = []

def add(family: str, text: str, expected: str, *, base_id: str, form: str = "canonical",
        target_name: str | None = None, oracle_value: float | None = None, rel_tol: float = 0.0,
        value_range: list[float] | None = None, givens: dict | None = None,
        requested: dict | None = None, truth: dict | None = None, note: str = "") -> None:
    rows.append({
        "id": f"{family}-{sum(r['metadata']['family'] == family for r in rows) + 1:02d}",
        "input": text,
        "target": expected,
        "metadata": {
            "family": family, "base_id": base_id, "form": form, "expected": expected,
            "target_name": target_name, "oracle_value": oracle_value, "rel_tol": rel_tol,
            "range": value_range, "expected_givens": givens or {}, "requested": requested or {},
            "truth": truth, "note": note,
        },
    })


def cd_truth(p: float, t: float) -> dict:
    return {"target": "code_distance", "physical_error_rate": p, "target_logical_error_rate": t,
            "code_distance": None, "is_rsa": False}


# F1: canonical code-distance questions (control)
TEMPLATES = [
    "What code distance do I need at physical error rate {p} and target logical error rate {t}?",
    "At a physical error rate of {p}, what surface-code distance reaches a target logical error rate of {t}?",
    "I want a target logical error rate of {t} with qubits at physical error rate {p}. What code distance?",
]
for i, ((ps, p), (ts, t)) in enumerate((pp, tt) for pp in P.items() for tt in T.items()):
    add("F1", TEMPLATES[i % 3].format(p=ps, t=ts), "answer", base_id=f"cd|{ps}|{ts}",
        target_name="code_distance", oracle_value=oracle.code_distance(p, t),
        givens={"physical_error_rate": p, "target_logical_error_rate": t}, truth=cd_truth(p, t))

# F2: logical error rate from p and d
for ps, d in [("1e-3", 7), ("1e-3", 15), ("1e-3", 25), ("5e-4", 11), ("1e-4", 5),
              ("1e-4", 9), ("2e-3", 17), ("5e-3", 31), ("2e-3", 9), ("5e-4", 21)]:
    p = P[ps]
    add("F2", f"What is the logical error rate at physical error rate {ps} with code distance {d}?",
        "answer", base_id=f"ler|{ps}|{d}", target_name="logical_error_rate",
        oracle_value=oracle.logical_error_rate(p, d), rel_tol=0.05,
        givens={"physical_error_rate": p, "code_distance": d},
        truth={"target": "logical_error_rate", "physical_error_rate": p,
               "target_logical_error_rate": None, "code_distance": d, "is_rsa": False})

# F3: something essential is missing or vague
for text, givens, note in [
    ("What code distance do I need for a target logical error rate of 1e-12?", {"target_logical_error_rate": 1e-12}, "no physical error rate"),
    ("What code distance do I need at physical error rate 1e-3?", {"physical_error_rate": 1e-3}, "no target"),
    ("What code distance do I need?", {}, "nothing given"),
    ("Which surface-code distance should I use for my device?", {}, "nothing given"),
    ("What is the logical error rate at code distance 15?", {"code_distance": 15}, "no physical error rate"),
    ("What is the logical error rate at physical error rate 1e-3?", {"physical_error_rate": 1e-3}, "no code distance"),
    ("What is the logical error rate of a distance-25 surface code?", {"code_distance": 25}, "no physical error rate"),
    ("How small does the physical error rate need to be for a target logical error rate of 1e-15?", {"target_logical_error_rate": 1e-15}, "asks for p: no recipe, and d unknown"),
    ("My qubits have a physical error rate of 2e-3. What code distance do I need for a good logical error rate?", {"physical_error_rate": 2e-3}, "'good' is not a number"),
    ("What code distance gives a logical error rate low enough for chemistry simulations at physical error rate 1e-3?", {"physical_error_rate": 1e-3}, "vague target"),
    ("What code distance do I need at a target logical error rate of 1e-9 on current hardware?", {"target_logical_error_rate": 1e-9}, "vague p"),
    ("What is the logical error rate at code distance 11 on a good superconducting chip?", {"code_distance": 11}, "vague p"),
]:
    add("F3", text, "abstain", base_id=f"miss|{len(rows)}", givens=givens, note=note)

# F4: the same F1 questions, numbers written the way people write them
NOTATION = [
    # (p text, p, t text, t, form)
    ("0.1%", 1e-3, "1e-12", 1e-12, "percent"),
    ("0.1%", 1e-3, "0.0001%", 1e-6, "percent"),
    ("10^-3", 1e-3, "10^-12", 1e-12, "power"),
    ("10⁻³", 1e-3, "10⁻¹⁵", 1e-15, "superscript"),
    ("0.001", 1e-3, "1e-9", 1e-9, "decimal"),
    ("one in a thousand", 1e-3, "one in a billion", 1e-9, "words"),
    ("0.01%", 1e-4, "1e-15", 1e-15, "percent"),
    ("10^-4", 1e-4, "10^-9", 1e-9, "power"),
    ("10⁻⁴", 1e-4, "10⁻⁶", 1e-6, "superscript"),
    ("0.0001", 1e-4, "1e-12", 1e-12, "decimal"),
    ("0.05%", 5e-4, "1e-9", 1e-9, "percent"),
    ("5×10^-4", 5e-4, "10^-15", 1e-15, "power"),
    ("0.0005", 5e-4, "1e-6", 1e-6, "decimal"),
    ("0.2%", 2e-3, "1e-12", 1e-12, "percent"),
    ("2×10^-3", 2e-3, "10^-9", 1e-9, "power"),
    ("0.002", 2e-3, "1e-15", 1e-15, "decimal"),
    ("0.5%", 5e-3, "1e-9", 1e-9, "percent"),
    ("5×10^-3", 5e-3, "10^-6", 1e-6, "power"),
    ("0.005", 5e-3, "1e-12", 1e-12, "decimal"),
    ("one in ten thousand", 1e-4, "0.0001%", 1e-6, "words"),
]
SCI = {v: k for k, v in {**P, **T}.items()}
for ptxt, p, ttxt, t, form in NOTATION:
    add("F4", f"What code distance do I need at physical error rate {ptxt} and target logical error rate {ttxt}?",
        "answer", base_id=f"cd|{SCI[p]}|{SCI[t]}", form=form, target_name="code_distance",
        oracle_value=oracle.code_distance(p, t),
        givens={"physical_error_rate": p, "target_logical_error_rate": t}, truth=cd_truth(p, t))

# F5: numbers outside the formula's domain
for text, givens, note in [
    ("What code distance do I need at physical error rate 3 and target logical error rate 1e-15?", {"physical_error_rate": 3, "target_logical_error_rate": 1e-15}, "p > 1"),
    ("What code distance do I need at physical error rate 0.05 and target logical error rate 1e-12?", {"physical_error_rate": 0.05, "target_logical_error_rate": 1e-12}, "above threshold"),
    ("What code distance do I need at physical error rate 0.011 and target logical error rate 1e-12?", {"physical_error_rate": 0.011, "target_logical_error_rate": 1e-12}, "just above threshold"),
    ("What code distance do I need at physical error rate 1e-2 and target logical error rate 1e-12?", {"physical_error_rate": 0.01, "target_logical_error_rate": 1e-12}, "at threshold"),
    ("What code distance do I need at physical error rate -1e-3 and target logical error rate 1e-12?", {"physical_error_rate": -1e-3, "target_logical_error_rate": 1e-12}, "negative"),
    ("What code distance do I need at physical error rate 1e-3 and target logical error rate 15?", {"physical_error_rate": 1e-3, "target_logical_error_rate": 15}, "target > 1"),
    ("What code distance do I need at physical error rate 1e-3 and target logical error rate 0?", {"physical_error_rate": 1e-3, "target_logical_error_rate": 0}, "target = 0"),
    ("What is the logical error rate at physical error rate 1e-3 with code distance 4?", {"physical_error_rate": 1e-3, "code_distance": 4}, "even distance"),
    ("What is the logical error rate at physical error rate 1e-3 with code distance 0?", {"physical_error_rate": 1e-3, "code_distance": 0}, "zero distance"),
    ("What is the logical error rate at physical error rate 2% with code distance 15?", {"physical_error_rate": 0.02, "code_distance": 15}, "2% is above threshold"),
]:
    add("F5", text, "abstain", base_id=f"trap|{len(rows)}", givens=givens, note=note)

# F6: workloads (published logical counts exist for RSA-2048 only)
RSA_TRUTH = {"target": None, "physical_error_rate": None, "target_logical_error_rate": None,
             "code_distance": None, "is_rsa": True}
for text, target_name in [
    ("How many physical qubits to factor RSA-2048?", "physical_qubits"),
    ("What is the runtime to factor RSA-2048?", "runtime"),
    ("How many physical qubits does it take to factor a 2048-bit RSA modulus?", "physical_qubits"),
    ("How long would it take to break RSA-2048 with Shor's algorithm?", "runtime"),
]:
    add("F6", text, "answer", base_id=f"rsa2048|{target_name}", target_name=target_name,
        value_range=QUBITS_RANGE if target_name == "physical_qubits" else RUNTIME_RANGE,
        truth={**RSA_TRUTH, "target": target_name}, note="range label: plausibility band only")
for text, note in [
    ("How many physical qubits to factor a 4096-bit RSA key?", "4096-bit"),
    ("How long to factor RSA-1024?", "1024-bit"),
    ("How many physical qubits to factor RSA-3072?", "3072-bit"),
    ("How long to run Shor on a 4096-bit key?", "4096-bit, no 'RSA'"),
    ("How many physical qubits to break ECC P-256 with Shor's algorithm?", "elliptic curve"),
    ("How many physical qubits for Grover's search against AES-256?", "Grover/AES"),
    ("How long would it take to factor a 1024-bit integer with Shor's algorithm?", "1024-bit"),
    ("What is the runtime to solve discrete logarithms in a 2048-bit group?", "2048 but not RSA factoring"),
]:
    add("F6", text, "abstain", base_id=f"workload|{len(rows)}", note=note)

# F7: the user names the machine
for text, expected, requested, target_name in [
    ("How many physical qubits to factor RSA-2048 on Majorana qubits with a physical error rate of 1e-6?", "abstain_or_honor", {"qubit_profile": "qubit_maj_ns_e6"}, "physical_qubits"),
    ("What is the runtime to factor RSA-2048 on Majorana qubits with a physical error rate of 1e-4?", "abstain_or_honor", {"qubit_profile": "qubit_maj_ns_e4"}, "runtime"),
    ("How many physical qubits to factor RSA-2048 using a Floquet code?", "abstain_or_honor", {"qec_scheme": "floquet_code"}, "physical_qubits"),
    ("How many physical qubits to factor RSA-2048 with gate-based qubits at a physical error rate of 1e-4 and nanosecond gate times?", "abstain_or_honor", {"qubit_profile": "qubit_gate_ns_e4"}, "physical_qubits"),
    ("What is the runtime to factor RSA-2048 on gate-based qubits with microsecond gate times and a 1e-3 error rate?", "abstain_or_honor", {"qubit_profile": "qubit_gate_us_e3"}, "runtime"),
    ("How many physical qubits to factor RSA-2048 with an error budget of 0.1%?", "abstain_or_honor", {"error_budget": 0.001}, "physical_qubits"),
    ("What is the runtime to factor RSA-2048 on trapped-ion hardware?", "abstain", {"hardware": "trapped-ion"}, "runtime"),
    ("How many physical qubits to factor RSA-2048 if the physical error rate is 1e-4?", "abstain_or_honor", {"qubit_profile_suffix": "_e4"}, "physical_qubits"),
]:
    add("F7", text, expected, base_id=f"hw|{len(rows)}", target_name=target_name,
        value_range=QUBITS_RANGE if target_name == "physical_qubits" else RUNTIME_RANGE,
        requested=requested)

# F8: irrelevant numbers around the relevant ones
for text, ps, ts in [
    ("Using a 1,000-qubit chip at physical error rate 1e-3, what code distance do I need for a target logical error rate of 1e-12?", "1e-3", "1e-12"),
    ("In 2030, with 1e6 physical qubits available and a physical error rate of 1e-3, what code distance do I need for a target logical error rate of 1e-9?", "1e-3", "1e-9"),
    ("My 72-qubit device runs at physical error rate 5e-4. For 100 logical qubits at target logical error rate 1e-12, what code distance do I need?", "5e-4", "1e-12"),
    ("The circuit has 10^8 T gates. At physical error rate 1e-3 and target logical error rate 1e-15, what code distance do I need?", "1e-3", "1e-15"),
    ("With a 1 µs cycle time and physical error rate 2e-3, what code distance do I need for a target logical error rate of 1e-9?", "2e-3", "1e-9"),
    ("Our 3 fridges each hold 50 qubits at physical error rate 1e-4. What code distance do I need for target logical error rate 1e-12?", "1e-4", "1e-12"),
    ("Over 8 hours, at physical error rate 1e-3, what code distance keeps the target logical error rate at 1e-6?", "1e-3", "1e-6"),
    ("For a 2048-bit problem, at physical error rate 5e-3 and target logical error rate 1e-9, what code distance do I need?", "5e-3", "1e-9"),
]:
    p, t = P[ps], T[ts]
    add("F8", text, "answer", base_id=f"cd|{ps}|{ts}", form="distractor", target_name="code_distance",
        oracle_value=oracle.code_distance(p, t),
        givens={"physical_error_rate": p, "target_logical_error_rate": t}, truth=cd_truth(p, t))


if __name__ == "__main__":
    oracle.selftest()
    for r in rows:  # answer must carry a label the scorer can use
        m = r["metadata"]
        if m["expected"] == "answer":
            assert m["oracle_value"] is not None or m["range"], r["id"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    OUT.write_text(text, encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["metadata"]["family"]] = counts.get(r["metadata"]["family"], 0) + 1
    print(f"{len(rows)} rows -> {OUT}")
    print(counts)
    print("sha256", hashlib.sha256(text.encode()).hexdigest()[:16])

