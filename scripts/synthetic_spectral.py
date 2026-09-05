"""
Synthetic spectral problems with known regime structure (generic, no doc hardcode).

Draws lognormal spectra (1e3 span), uniform unitary/radial, assigns regimes via
spec3/gap1/upsilon thresholds calibrated to target occupancy r, builds ground-truth
f* satisfying regime-conditional laws + Gaussian noise. For scaling sweeps and
falsification runs. Deterministic via seed.
"""
import numpy as np


def generate(n=1000, r=(0.5, 0.5, 0.5), noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    # Heavy-tailed spectra spanning 1e3
    lam = rng.lognormal(mean=0.0, sigma=1.5, size=(n, 22)).astype(np.float64)
    lam = np.sort(lam, axis=1)[:, ::-1]
    # Rescale rows to span ~1e3 max/min ratio guard
    ups = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    rho = rng.uniform(0.0, 5.0, size=n).astype(np.float64)
    spec3 = lam[:, :3].sum(axis=1)
    gap1 = lam[:, 0] - lam[:, 1]
    # Thresholds calibrated to target occupancy via quantiles (generic)
    ctop = float(np.quantile(spec3, 1.0 - r[0]))
    cgap = float(np.quantile(gap1, 1.0 - r[1]))
    cups = float(np.quantile(ups ** 2, 1.0 - r[2]))
    top = spec3 > ctop
    gap = gap1 > cgap
    ups_on = (ups ** 2) > cups
    # Ground truth: monotone leading + supermodular pair + range tightening (simplified)
    fstar = 0.5 * spec3 + 0.3 * np.maximum(0.0, gap1) + 0.2 * (1.0 - np.abs(ups)) * spec3 / (spec3.mean() + 1e-8)
    # Radial unimodal peak at 2.5
    fstar = fstar - 0.1 * (rho - 2.5) ** 2
    y = fstar + rng.normal(0.0, noise, size=n)
    regimes = np.stack([top, gap, ups_on], axis=1).astype(np.int64)
    return {"lam": lam.astype(np.float32), "ups": ups.astype(np.float32), "rho": rho.astype(np.float32),
            "y": y.astype(np.float32), "regimes": regimes,
            "thresholds": {"ctop": ctop, "cgap": cgap, "cups": cups}}


if __name__ == "__main__":
    d = generate(n=100)
    print(f"n=100 regimes mean={d['regimes'].mean(axis=0).round(2).tolist()} thresholds={d['thresholds']}")
