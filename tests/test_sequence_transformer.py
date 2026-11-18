import numpy as np
import pandas as pd

from pitchsim.sequence.transformer import TOKENS, TransformerSeq


def _abs(n_ab: int, pitcher: int, rng) -> pd.DataFrame:
    """Synthetic at-bats: pitcher 1 alternates FF, SL, FF...; pitcher 2 throws only CH.
    Context columns are constant, so the transformer must use pitcher + history."""
    rows = []
    for ab in range(n_ab):
        for k in range(rng.integers(2, 6)):
            t = "CH" if pitcher == 2 else ("FF" if k % 2 == 0 else "SL")
            prev = "start" if k == 0 else ("CH" if pitcher == 2 else ("FF" if k % 2 == 1 else "SL"))
            rows.append({"pitcher": pitcher, "game_pk": ab // 5, "at_bat_number": ab, "pitch_number": k + 1,
                         "count": "0-0", "stand": "R", "tto": "1", "prev_type": prev,
                         "prev_result": "ball", "prev_zone": "5", "pitch_type": t})
    return pd.DataFrame(rows)


def test_transformer_learns_alternation_and_keeps_row_order():
    rng = np.random.default_rng(0)
    df = pd.concat([_abs(200, 1, rng), _abs(100, 2, rng)], ignore_index=True)
    m = TransformerSeq(["CH", "FF", "SL"], max_epochs=15, grid=[(16, 0.0, 0.0)]).fit(df)
    shuffled = df.sample(frac=1, random_state=1)                 # predict must follow df's row order
    p = m.predict(shuffled)
    truth = shuffled["pitch_type"].map({"CH": 0, "FF": 1, "SL": 2}).to_numpy()
    assert np.allclose(p.sum(1), 1)
    assert (p.argmax(1) == truth).mean() > 0.95
    # pitcher 1 never threw CH in training: masked to the 1e-4 floor
    assert p[shuffled["pitcher"].to_numpy() == 1, 0].max() < 1e-3
    assert set(TOKENS) <= set(df)
