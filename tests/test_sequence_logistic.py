import numpy as np
import pandas as pd

from pitchsim.sequence.context import add_context
from pitchsim.sequence.logistic import LogisticSeq


def test_prev_type_resets_per_at_bat_and_breaks_on_untyped():
    df = pd.DataFrame({"game_pk": [1] * 5, "at_bat_number": [1, 1, 1, 2, 2],
                       "pitch_number": [2, 1, 3, 1, 2], "balls": [0] * 5, "strikes": [1, 0, 2, 0, 1],
                       "pitch_type": ["SL", "FF", None, "CH", "FF"]})
    d = add_context(df)
    assert d["prev_type"].tolist() == ["start", "FF", "SL", "start", "CH"]
    assert d["count"].tolist()[:3] == ["0-0", "0-1", "0-2"]


def test_logistic_learns_previous_pitch_and_samples():
    rng = np.random.default_rng(0)
    n = 600   # synthetic: always throws SL after FF, FF otherwise
    prev = rng.choice(["start", "FF", "SL"], n)
    df = pd.DataFrame({"count": "0-0", "stand": "R", "prev_type": prev,
                       "pitch_type": np.where(prev == "FF", "SL", "FF")})
    m = LogisticSeq(["CH", "FF", "SL"], C=10).fit(df)
    p = m.predict(pd.DataFrame({"count": ["0-0"] * 2, "stand": ["R"] * 2, "prev_type": ["FF", "SL"]}))
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()   # CH never seen, still nonzero
    assert p[0, 2] > 0.9 and p[1, 1] > 0.9
    assert set(m.sample(df.head(5), rng)) <= {"CH", "FF", "SL"}
