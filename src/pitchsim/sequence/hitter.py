"""M7 "average hitter": P(take / whiff / foul / in play | count, handedness, pitch type, location).

Learned per pitcher from that pitcher's real Statcast pitch results (spec goal 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

OUTCOMES = ["take", "whiff", "foul", "in_play"]
_MAP = {"ball": "take", "blocked_ball": "take", "pitchout": "take", "called_strike": "take",
        "automatic_strike": "take", "automatic_ball": "take",
        "swinging_strike": "whiff", "swinging_strike_blocked": "whiff", "missed_bunt": "whiff",
        "foul_tip": "whiff",   # a caught foul tip is a strike that can end the AB
        "foul": "foul", "foul_bunt": "foul", "hit_into_play": "in_play"}
FEATURES = ["count", "stand", "pitch_type", "zone_bin"]
BINS = ["heart", "edge_in", "edge_out", "chase", "waste"]
HALF_W_FT = 17 / 24 + 1.45 / 12   # plate half-width + ball radius: any overlap with the plate is "in"
BALL_R_FT = 1.45 / 12


def outcome_of(description: str) -> str | None:
    """Statcast `description` -> one of OUTCOMES, or None (hit_by_pitch etc., dropped in M7)."""
    return _MAP.get(description)


def zone_bin(x_ft, z_ft, sz_top_ft, sz_bot_ft):
    """Bin a pitch by signed distance (inches, negative inside) from the edge of the strike zone
    grown by one ball radius: heart < -3, edge_in -3..0, edge_out 0..3, chase 3..9, waste > 9.

    Why: swing/contact behaviour is driven mostly by how close the pitch is to the zone edge
    (the chase / zone-contact splits on Baseball Savant), so a few
    distance bins capture it without a 2D smooth. Vectorised; scalars return a str.
    """
    x, z = np.asarray(x_ft, float), np.asarray(z_ft, float)
    dx = np.abs(x) - HALF_W_FT
    dz = np.maximum(np.asarray(sz_bot_ft) - BALL_R_FT - z, z - (np.asarray(sz_top_ft) + BALL_R_FT))
    inside = np.maximum(dx, dz)                                  # < 0 inside the box
    outside = np.hypot(np.maximum(dx, 0), np.maximum(dz, 0))
    d_in = np.where((dx < 0) & (dz < 0), inside, outside) * 12
    out = np.asarray(BINS)[np.digitize(d_in, [-3, 0, 3, 9], right=True)]
    return str(out) if out.ndim == 0 else pd.Series(out, index=getattr(x_ft, "index", None))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known outcome, location and zone height; adds `zone_bin` and `outcome`."""
    d = df.assign(outcome=df["description"].map(outcome_of))
    d = d.dropna(subset=["outcome", "x_ft", "z_ft", "sz_top", "sz_bot", "count", "stand", "pitch_type"])
    return d.assign(zone_bin=zone_bin(d["x_ft"], d["z_ft"], d["sz_top"], d["sz_bot"]))


def _y(d: pd.DataFrame) -> np.ndarray:
    return d["outcome"].map(OUTCOMES.index).to_numpy()


class HitterModel:
    """Softmax logistic regression on one-hot (count, stand, pitch_type, zone_bin).

    Same idea as LogisticSeq: each level shifts the log-odds of each outcome, effects add
    (Hastie, Tibshirani & Friedman, ESL 4.4). C is the inverse L2 strength.
    """

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, df: pd.DataFrame) -> "HitterModel":
        d = prepare(df)
        self.enc = OneHotEncoder(handle_unknown="ignore").fit(d[FEATURES])
        lr = LogisticRegression(C=self.C, max_iter=2000).fit(self.enc.transform(d[FEATURES]), _y(d))
        k, nf = len(OUTCOMES), self.enc.transform(d[FEATURES][:1]).shape[1]
        self.coef, self.intercept = np.zeros((k, nf)), np.full(k, np.log(1e-4))   # class absent from training
        w, b = lr.coef_, lr.intercept_
        if len(lr.classes_) == 2:                  # sklearn gives one row; softmax(0, wx+b) == sigmoid
            w, b = np.vstack([np.zeros_like(w), w]), np.array([0.0, b[0]])
        self.coef[lr.classes_], self.intercept[lr.classes_] = w, b
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 4) probabilities in OUTCOMES order; df needs `zone_bin` or x_ft/z_ft/sz_top/sz_bot."""
        d = df if "zone_bin" in df else df.assign(zone_bin=zone_bin(df["x_ft"], df["z_ft"], df["sz_top"], df["sz_bot"]))
        z = self.enc.transform(d[FEATURES]) @ self.coef.T + self.intercept
        p = np.exp(z - z.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def to_json(self) -> dict:
        """Enough to recompute predict: p = softmax(onehot(features) @ coef.T + intercept)."""
        return {"features": FEATURES, "categories": [[str(v) for v in c] for c in self.enc.categories_],
                "classes": OUTCOMES, "coef": self.coef.tolist(), "intercept": self.intercept.tolist()}


class HitterTable:
    """P(outcome | count, zone_bin) = (n_cell + a * p_overall) / (N_cell + a), as CountBaseline:
    a Dirichlet prior centred on the pitcher's overall outcome mix with `a` pseudo-pitches."""

    def __init__(self, a: float = 1.0):
        self.a = a

    def fit(self, df: pd.DataFrame) -> "HitterTable":
        d = prepare(df)
        n = pd.crosstab([d["count"], d["zone_bin"]], d["outcome"]).reindex(columns=OUTCOMES, fill_value=0)
        o = d["outcome"].value_counts().reindex(OUTCOMES, fill_value=0).to_numpy() + 1.0
        self.overall = o / o.sum()
        self.cells = {k: r.to_numpy() for k, r in n.iterrows()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d = df if "zone_bin" in df else df.assign(zone_bin=zone_bin(df["x_ft"], df["z_ft"], df["sz_top"], df["sz_bot"]))
        out, zero = np.empty((len(d), len(OUTCOMES))), np.zeros(len(OUTCOMES))
        for i, k in enumerate(zip(d["count"], d["zone_bin"])):
            c = self.cells.get(k, zero)
            out[i] = (c + self.a * self.overall) / (c.sum() + self.a)
        return out

    def to_json(self) -> dict:
        return {"a": self.a, "classes": OUTCOMES, "overall": self.overall.tolist(),
                "cells": {f"{c}|{b}": v.tolist() for (c, b), v in self.cells.items()}}
