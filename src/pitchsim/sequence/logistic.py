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

    def __init__(self, types: list[str], C: float = 1.0, features: list[str] = FEATURES):
        self.types, self.C, self.features = list(types), C, list(features)

    def fit(self, train: pd.DataFrame) -> "LogisticSeq":
        self.model = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                   LogisticRegression(C=self.C, max_iter=2000))
        self.model.fit(train[self.features], train["pitch_type"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(len(df), n_types) probabilities over self.types; rows sum to 1."""
        out = np.zeros((len(df), len(self.types)))
        col = [self.types.index(c) for c in self.model.classes_]
        out[:, col] = self.model.predict_proba(df[self.features])
        # ponytail: a type absent from this fold's training games gets a 1e-4 floor; a real
        # prior (like CountBaseline's +1) if rare pitch types ever matter.
        out = np.maximum(out, 1e-4)
        return out / out.sum(axis=1, keepdims=True)

    def sample(self, context: pd.DataFrame, rng: np.random.Generator) -> list[str]:
        """One pitch type per context row, drawn from the distribution (never argmax)."""
        return [str(rng.choice(self.types, p=p)) for p in self.predict(context)]


class BoostedSeq(LogisticSeq):
    """Gradient-boosted trees over the same categorical context (Friedman 2001; scikit-learn's
    histogram version, the same algorithm family as XGBoost/LightGBM).

    Unlike the logistic model, a tree splits on one feature and then another, so it can learn
    interactions like "2 strikes AND lefty -> more changeups" without a hand-made cross feature.
    The price is more parameters: with a few thousand pitches it can overfit, which is why the
    trees are shallow, leaves need min_samples_leaf pitches, and the number of trees is the
    hyperparameter tuned on held-out training games. `C` here is that number of boosting rounds.
    """

    def fit(self, train: pd.DataFrame) -> "BoostedSeq":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder
        self.model = make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            HistGradientBoostingClassifier(categorical_features=[True] * len(self.features),
                                           learning_rate=0.05, max_iter=int(self.C), max_depth=3,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           early_stopping=False, random_state=0))
        self.model.fit(train[self.features], train["pitch_type"])
        return self
