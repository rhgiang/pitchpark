"""Tests for the location model. All data here is SYNTHETIC (generated in-test), not Statcast."""
import numpy as np
import pandas as pd

from pitchsim.sequence.location import LocationModel


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"count": rng.choice(["0-0", "0-2"], n), "stand": rng.choice(["L", "R"], n),
                      "prev_type": rng.choice(["FF", "SL"], n)})
    cov = np.array([[0.30, 0.05], [0.05, 0.20]])
    shift = np.where(d["count"] == "0-2", -0.6, 0.0)          # known effect: lower with 2 strikes
    noise = rng.multivariate_normal([0, 0], cov, size=n)
    d["x_ft"], d["z_ft"] = 0.1 + noise[:, 0], 2.5 + shift + noise[:, 1]
    return d, cov


def test_recovers_mean_shift_and_samples_match():
    d, cov = _synthetic()
    m = LocationModel(alpha=1.0).fit(d)
    two = d[d["count"] == "0-2"].head(1); zero = d[d["count"] == "0-0"].head(1)
    assert abs((m.mean(two) - m.mean(zero))[0, 1] - (-0.6)) < 0.05
    assert abs((m.mean(two) - m.mean(zero))[0, 0]) < 0.05
    ctx = pd.concat([zero] * 20000)
    s = m.sample(ctx, np.random.default_rng(1))
    assert np.allclose(s.mean(axis=0), m.mean(zero)[0], atol=0.02)
    assert np.allclose(np.cov(s.T), cov, atol=0.02)
    # ~50% / 90% of fresh draws fall inside the central ellipses
    test = ctx.assign(x_ft=s[:, 0], z_ft=s[:, 1])
    assert abs(m.inside(test, 0.5).mean() - 0.5) < 0.02 and abs(m.inside(test, 0.9).mean() - 0.9) < 0.02


def test_conditioned_beats_reference_when_shift_is_real():
    d, _ = _synthetic()
    tr, te = d.iloc[:3000], d.iloc[3000:]
    ref = LocationModel(features=[]).fit(tr).logpdf(te).mean()
    assert LocationModel(alpha=1.0).fit(tr).logpdf(te).mean() > ref + 0.05
