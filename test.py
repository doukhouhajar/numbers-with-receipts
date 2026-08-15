from groundwork.agent import run
from groundwork.verify import check_cross_method

a = run("How many physical qubits to factor RSA-2048?")
q = a.derivation.quantities
print("code_distance grounded:", "code_distance" in q and q["code_distance"].is_grounded, "=", q["code_distance"].value)
print(check_cross_method("physical_qubits", q, a.result))