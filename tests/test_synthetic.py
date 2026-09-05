"""Synthetic generator regime occupancy + shapes."""
from scripts.synthetic_spectral import generate


def test_shapes_and_occupancy():
    d = generate(n=200, r=(0.5, 0.5, 0.5), seed=1)
    assert d["lam"].shape == (200, 22)
    assert d["y"].shape == (200,)
    assert d["regimes"].shape == (200, 3)
    # Occupancy near target within tolerance (quantile calibration)
    means = d["regimes"].mean(axis=0)
    for m in means:
        assert 0.3 <= float(m) <= 0.7
