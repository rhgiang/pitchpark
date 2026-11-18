"""Context features for pitch-selection models, computed from raw Statcast rows."""
from __future__ import annotations

import pandas as pd

START = "start"   # previous-pitch value for the first pitch of an at-bat


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df, in pitch order, with `prev_type` (the pitch before this one in the same at-bat,
    START for the first) and `count` ("b-s"). Untyped pitches stay in so they still break the
    chain: the pitch after one gets prev_type "unknown", not the pitch two back.
    """
    d = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    prev = d.groupby(["game_pk", "at_bat_number"])["pitch_type"].shift(1)
    first = d["pitch_number"] == d.groupby(["game_pk", "at_bat_number"])["pitch_number"].transform("min")
    d["prev_type"] = prev.fillna("unknown").where(~first, START)
    d["count"] = d["balls"].astype(int).astype(str) + "-" + d["strikes"].astype(int).astype(str)
    return d
