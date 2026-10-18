from __future__ import annotations

# groundtruth for eval, without importing groundwork. If the oracle shared code with the system, it would share its bugs

A = 0.03
P_TH = 0.01
D_MAX = 2001


def logical_error_rate(p: float, d: int) -> float | None:
    if not (0 < p < P_TH) or d < 1 or d % 2 == 0:
        return None
    return A * (p / P_TH) ** ((d + 1) / 2)


def code_distance(p: float, target: float) -> int | None:
    if not (0 < p < P_TH) or not (0 < target < 1):
        return None
    for d in range(1, D_MAX, 2):
        p_l = logical_error_rate(p, d)
        if p_l is not None and p_l <= target:
            return d
    return None


# if any of these fail the oracle is wrong
HAND_CHECKED = {
    (1e-3, 1e-15): 27,
    (1e-3, 1e-12): 21,
    (1e-3, 1e-6): 9,
    (1e-4, 1e-15): 13,
    (5e-3, 1e-9): 49,
}

def selftest() -> None:
    for (p, t), d in HAND_CHECKED.items():
        got = code_distance(p, t)
        assert got == d, f"oracle code_distance({p}, {t}) = {got}, hand value {d}"
    assert code_distance(0.011, 1e-15) is None
    assert logical_error_rate(1e-3, 4) is None


if __name__ == "__main__":
    selftest()
    print("oracle ok")
