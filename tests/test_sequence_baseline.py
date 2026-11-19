import numpy as np
import pandas as pd

from pitchsim.sequence.baseline import CountBaseline, log_loss


def test_baseline_probabilities_and_smoothing():
    df = pd.DataFrame({"balls": [0, 0, 0, 3], "strikes": [0, 0, 0, 0], "pitch_type": ["FF", "FF", "SL", "FF"]})
    m = CountBaseline(["FF", "SL", "CH"], a=2.0).fit(df)
    p = m.predict(pd.DataFrame({"balls": [0, 3, 2], "strikes": [0, 0, 2]}))  # last: unseen count
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()   # no zero mass, even for unseen CH
    assert p[0, 0] > p[0, 1] > p[0, 2]
    assert np.allclose(p[2], m.overall)                       # unseen count falls back to the overall mix
    assert log_loss(p, np.array([0, 0, 0])) > 0
    assert m.sample(0, 0, np.random.default_rng(0)) in {"FF", "SL", "CH"}
