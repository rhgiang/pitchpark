"""M6 step 2: multinomial logistic regression on count, batter side and previous pitch."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["count", "stand", "prev_type"]


class LogisticSeq:
    """P(type | context) = softmax(W x + b), x = one-hot count, stand and previous pitch type.

    Multinomial (softmax) logistic regression: each feature level adds a log-odds shift per pitch
    type (Hastie, Tibshirani & Friedman, ESL 4.4). Effects are additive in log-odds, so e.g. the
    "2 strikes" and "vs lefty" shifts combine without a separate cell for every combination - that
    pooling is what lets it use far more context than the count table. No interactions yet.
    C is the inverse L2 strength: small C shrinks rare levels toward the overall mix, like `a` in
    CountBaseline.
    """

    def __init__(self, types: list[str], C: float = 1.0):
        self.types, self.C = list(types), C

    def fit(self, train: pd.DataFrame) -> "LogisticSeq":
        self.model = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                   LogisticRegression(C=self.C, max_iter=2000))
        self.model.fit(train[FEATURES], train["pitch_type"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(len(df), n_types) probabilities over self.types; rows sum to 1."""
        out = np.zeros((len(df), len(self.types)))
        col = [self.types.index(c) for c in self.model.classes_]
        out[:, col] = self.model.predict_proba(df[FEATURES])
        # ponytail: a type absent from this fold's training games gets a 1e-4 floor; a real
        # prior (like CountBaseline's +1) if rare pitch types ever matter.
        out = np.maximum(out, 1e-4)
        return out / out.sum(axis=1, keepdims=True)

    def sample(self, context: pd.DataFrame, rng: np.random.Generator) -> list[str]:
        """One pitch type per context row, drawn from the distribution (never argmax)."""
        return [str(rng.choice(self.types, p=p)) for p in self.predict(context)]
