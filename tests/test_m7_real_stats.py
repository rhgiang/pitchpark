"""m7_real_stats on a tiny SYNTHETIC 3-game frame (hand-built in this test, not real data)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m7_real_stats import real_stats  # noqa: E402

ZONE = {"L": {"top_ft": 3.5, "bot_ft": 1.5}, "R": {"top_ft": 3.5, "bot_ft": 1.5}}


def _ab(game, n, stand, pitches, events):
    """pitches: [(type, description, x_ft, z_ft)] ; counts derived by hand-simple tracking."""
    rows, b, s = [], 0, 0
    for i, (t, d, x, z) in enumerate(pitches, 1):
        rows.append(dict(game_pk=game, at_bat_number=n, pitch_number=i, stand=stand, pitch_type=t,
                         description=d, events=events if i == len(pitches) else None,
                         balls=b, strikes=s, count=f"{b}-{s}", x_ft=x, z_ft=z))
        if d in ("ball",): b += 1
        elif d in ("called_strike", "swinging_strike") or (d == "foul" and s < 2): s += 1
    return rows


def frame():
    r = []
    # game 1: R K in 3 (called strike in zone, swing, swing); R walk in 4 (all balls off the plate)
    r += _ab(1, 1, "R", [("FF", "called_strike", 0, 2.5), ("SL", "swinging_strike", 0, 2.5),
                         ("SL", "swinging_strike", 0, 2.5)], "strikeout")
    r += _ab(1, 2, "R", [("FF", "ball", 2, 2.5)] * 4, "walk")
    # game 2: L in play on first pitch; L HBP (excluded); L truncated (excluded)
    r += _ab(2, 1, "L", [("FF", "hit_into_play", 0, 2.5)], "field_out")
    r += _ab(2, 2, "L", [("FF", "hit_by_pitch", 0, 2.5)], "hit_by_pitch")
    r += _ab(2, 3, "L", [("FF", "ball", 2, 2.5)], "caught_stealing_2b")
    # game 3: R foul then in play
    r += _ab(3, 1, "R", [("FF", "foul", 0, 2.5), ("CH", "hit_into_play", 0, 2.5)], "single")
    return pd.DataFrame(r)


def test_hand_counts():
    out = real_stats(frame(), ["FF", "SL", "CH"], ZONE, n_boot=200, seed=0)
    assert out["excluded_abs"] == {"hbp": 1, "truncated": 1, "intentional_walk": 0, "missing_data": 0}
    r = out["R"]
    assert r["n_ab"] == 3 and out["L"]["n_ab"] == 1
    assert r["pitches_per_ab"]["est"] == (3 + 4 + 2) / 3
    assert {k: v["est"] for k, v in r["outcomes"].items()} == \
        {"strikeout": 1 / 3, "walk": 1 / 3, "in_play": 1 / 3}
    assert r["first_pitch_strike"]["est"] == 2 / 3     # called strike, foul; the walk AB started with a ball
    assert r["zone_rate"]["est"] == 5 / 9              # zone pitches 3 (K ab) + 0 + 2 (foul/in play)


def test_intervals_contain_estimate():
    out = real_stats(frame(), ["FF", "SL", "CH"], ZONE, n_boot=200, seed=0)
    for s in "LR":
        for st in [out[s]["zone_rate"], out[s]["first_pitch_strike"], out[s]["pitches_per_ab"],
                   *out[s]["outcomes"].values()]:
            assert st["lo"] <= st["est"] <= st["hi"]
    for s in "LR":
        assert out["stand_share"][s]["lo"] <= out["stand_share"][s]["est"] <= out["stand_share"][s]["hi"]
