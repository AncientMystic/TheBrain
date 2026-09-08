"""Fidelity gate proof (phase 99)."""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def test_fidelity_gate():
    from core.entity_linking import fidelity_gate as F
    ok, bits, top = F([0.9, 0.05, 0.05])
    assert ok is True and bits > 1.0 and top == 0.9, (ok, bits, top)
    ok, bits, top = F([0.34, 0.33, 0.33])
    assert ok is False and bits < 0.1, (ok, bits, top)
    assert F([]) == (False, 0.0, 0.0)
    assert F([1.0])[0] is True
    assert F([0, 0, 0]) == (False, 0.0, 0.0)
    print("test_fidelity_gate: OK (peaked accepts, flat abstains, degenerate safe)")


if __name__ == "__main__":
    test_fidelity_gate()
