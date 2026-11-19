"""M6 baseline: pitch-type frequency by count, smoothed toward the pitcher's overall mix."""
from __future__ import annotations

import numpy as np
import pandas as pd


class CountBaseline:
    """P(pitch_type | balls, strikes) = (n_cell + a * p_overall) / (N_cell + a).

    The count cells are small (a 3-0 count is rare), so a raw frequency would put p = 0 on types
    never seen there and blow up log loss. Smoothing is a Dirichlet prior centred on the overall
    mix with strength `a` pseudo-pitches (Laplace-style; Murphy, "Machine Learning", 3.3): a cell
    with few pitches stays near the overall mix, a big cell follows its own data.
    """

    def __init__(self, types: list[str], a: float = 10.0):
        self.types, self.a = list(types), a

    def fit(self, train: pd.DataFrame) -> "CountBaseline":
        t = train.dropna(subset=["pitch_type", "balls", "strikes"])
        n = pd.crosstab([t["balls"], t["strikes"]], t["pitch_type"]).reindex(columns=self.types, fill_value=0)
        # +1 on the overall mix so a type unseen in training still gets a little mass
        overall = t["pitch_type"].value_counts().reindex(self.types, fill_value=0).to_numpy() + 1.0
        self.overall = overall / overall.sum()
        self.cells = {k: row.to_numpy() for k, row in n.iterrows()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(len(df), n_types) probabilities; rows sum to 1."""
        out = np.empty((len(df), len(self.types)))
        zero = np.zeros(len(self.types))
        for i, k in enumerate(zip(df["balls"], df["strikes"])):
            c = self.cells.get(k, zero)
            out[i] = (c + self.a * self.overall) / (c.sum() + self.a)
        return out

    def sample(self, balls: int, strikes: int, rng: np.random.Generator) -> str:
        """Draw one pitch type (never argmax: the sim must keep the pitcher's variety)."""
        p = self.predict(pd.DataFrame({"balls": [balls], "strikes": [strikes]}))[0]
        return str(rng.choice(self.types, p=p))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean -log p[true type]; y is the integer type index per row."""
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Pooled one-vs-rest calibration: every (pitch, type) pair is one forecast.
    Per probability bin: mean predicted vs observed frequency. A calibrated model has them equal."""
    flat_p = p.ravel()
    hit = (np.arange(p.shape[1])[None, :] == y[:, None]).ravel()
    b = np.minimum((flat_p * bins).astype(int), bins - 1)
    g = pd.DataFrame({"b": b, "p": flat_p, "hit": hit}).groupby("b")
    return pd.DataFrame({"n": g.size(), "mean_pred": g["p"].mean(), "observed": g["hit"].mean()})
