import numpy as np
import pandas as pd

from pitchsim.sequence.hitter import OUTCOMES, HitterModel, HitterTable, outcome_of, zone_bin


def synthetic_hitter_frame(n: int, seed: int) -> pd.DataFrame:
    """SYNTHETIC (not real Statcast): random locations; pitches near the middle are swung at
    (whiff/foul/in_play) 85% of the time, far-out ones 10%."""
    rng = np.random.default_rng(seed)
    x, z = rng.normal(0, 0.9, n), rng.normal(2.5, 0.9, n)
    df = pd.DataFrame({"count": rng.choice(["0-0", "1-1", "0-2"], n), "stand": rng.choice(["L", "R"], n),
                       "pitch_type": rng.choice(["FF", "SL"], n), "x_ft": x, "z_ft": z,
                       "sz_top": 3.5, "sz_bot": 1.6})
    heart = (zone_bin(df.x_ft, df.z_ft, df.sz_top, df.sz_bot) == "heart").to_numpy()
    swing = rng.random(n) < np.where(heart, 0.85, 0.10)
    df["description"] = np.where(swing, rng.choice(["swinging_strike", "foul", "hit_into_play"], n), "ball")
    return df


def test_outcome_mapping():
    assert outcome_of("ball") == "take" and outcome_of("called_strike") == "take"
    assert outcome_of("blocked_ball") == "take" and outcome_of("pitchout") == "take"
    assert outcome_of("swinging_strike") == "whiff" and outcome_of("swinging_strike_blocked") == "whiff"
    assert outcome_of("foul_tip") == "whiff"      # a caught foul tip is a strike that can end the AB
    assert outcome_of("missed_bunt") == "whiff"
    assert outcome_of("foul") == "foul" and outcome_of("foul_bunt") == "foul"
    assert outcome_of("hit_into_play") == "in_play"
    assert outcome_of("hit_by_pitch") is None     # dropped: no HBP in M7 (spec non-goal)


def test_zone_bins():
    top, bot = 3.5, 1.6
    assert zone_bin(0.0, 2.5, top, bot) == "heart"
    assert zone_bin(0.70, 2.5, top, bot) == "edge_in"     # 0.13 ft inside the 0.83 ft half-width
    assert zone_bin(0.95, 2.5, top, bot) == "edge_out"
    assert zone_bin(1.3, 2.5, top, bot) == "chase"
    assert zone_bin(2.5, 2.5, top, bot) == "waste"


def test_predict_rows_sum_to_one_and_fit_signal():
    df = synthetic_hitter_frame(n=4000, seed=0)
    heart = (zone_bin(df.x_ft, df.z_ft, df.sz_top, df.sz_bot) == "heart").to_numpy()
    for m in (HitterModel(), HitterTable()):
        p = m.fit(df).predict(df)
        assert p.shape == (len(df), len(OUTCOMES)) and np.allclose(p.sum(1), 1)
        assert p[heart, 0].mean() < p[~heart, 0].mean()  # fewer takes in the heart


def test_model_to_json_recomputes_predict():
    df = synthetic_hitter_frame(n=1000, seed=1)
    m = HitterModel().fit(df)
    j = m.to_json()
    assert j["classes"] == OUTCOMES
    cols = ["count", "stand", "pitch_type", "zone_bin"]
    d = df.assign(zone_bin=zone_bin(df.x_ft, df.z_ft, df.sz_top, df.sz_bot))
    x = np.zeros((len(d), sum(len(c) for c in j["categories"])))
    off = 0
    for k, c in enumerate(j["categories"]):
        for i, v in enumerate(c):
            x[:, off + i] = (d[cols[k]] == v).to_numpy()
        off += len(c)
    z = x @ np.array(j["coef"]).T + np.array(j["intercept"])
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    assert j["features"] == cols and np.allclose(p, m.predict(df))
