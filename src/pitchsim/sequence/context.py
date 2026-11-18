"""Context features for pitch-selection models, computed from raw Statcast rows."""
from __future__ import annotations

import pandas as pd

START = "start"   # previous-pitch value for the first pitch of an at-bat

# Statcast `description` -> what the batter saw/did, coarse enough to have data in every bucket.
# Balls in play and HBP end the at-bat, so they never appear as a previous pitch.
RESULT = {"ball": "ball", "blocked_ball": "ball", "pitchout": "ball",
          "called_strike": "called", "automatic_strike": "called",
          "swinging_strike": "whiff", "swinging_strike_blocked": "whiff", "missed_bunt": "whiff",
          "foul": "foul", "foul_tip": "foul", "foul_bunt": "foul"}


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df, in pitch order, with `prev_type` (the pitch before this one in the same at-bat,
    START for the first) and `count` ("b-s"). Untyped pitches stay in so they still break the
    chain: the pitch after one gets prev_type "unknown", not the pitch two back.
    """
    d = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    prev = d.groupby(["game_pk", "at_bat_number"])["pitch_type"].shift(1)
    first = d["pitch_number"] == d.groupby(["game_pk", "at_bat_number"])["pitch_number"].transform("min")
    d["prev_type"] = prev.fillna("unknown").where(~first, START)
    ab = d.groupby(["game_pk", "at_bat_number"])
    d["prev_result"] = ab["description"].shift(1).map(RESULT).fillna("other").where(~first, START)
    # Statcast zone: 1-9 = the 3x3 strike-zone grid, 11-14 = the four outside quadrants
    d["prev_zone"] = ab["zone"].shift(1).astype("Int64").astype(str).fillna("unknown").where(~first, START)
    d["count"] = d["balls"].astype(int).astype(str) + "-" + d["strikes"].astype(int).astype(str)
    # The rest are categorical bins so the one-hot logistic can bend freely without a functional
    # form. Bin edges are judgement calls, not tuned.
    # Fatigue: the pitcher's pitch number in this game (he is the only pitcher in these CSVs).
    n = d.groupby("game_pk").cumcount()
    d["pitch_of_game"] = pd.cut(n, [-1, 24, 49, 74, 99, 10_000], labels=["0-24", "25-49", "50-74", "75-99", "100+"]).astype(str)
    d["tto"] = d["n_thruorder_pitcher"].clip(upper=3).astype("Int64").astype(str)   # 3 = third time or later
    d["outs"] = d["outs_when_up"].astype(int).astype(str)
    d["runners"] = (d[["on_1b", "on_2b", "on_3b"]].notna().astype(int).astype(str)).agg("".join, axis=1)  # "101" = 1st & 3rd
    d["score_diff"] = (d["fld_score"] - d["bat_score"]).clip(-3, 3).astype(int).astype(str)          # + = pitcher's team ahead
    return d
