"""M6 target location: where does a pitch of a given type cross the plate, given the context?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, multivariate_normal
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["count", "stand", "prev_type"]   # same context as the pitch-type logistic model


class LocationModel:
    """Location (x, z) in feet of ONE pitch type: a 2D Gaussian whose mean shifts with context.

    mean(context) = b + W x, x = one-hot count, stand and previous pitch type, fitted by ridge
    regression (L2 strength `alpha`; large alpha shrinks every shift to 0, so the model falls back
    to the plain Gaussian). The 2x2 covariance is the residual covariance, the same for every
    context (a "mean shift only" model: a pitcher's aim moves with the count, but we assume the
    scatter around the aim does not). With features=[] this is the unconditional reference.
    Ridge / Gaussian-noise regression: Hastie, Tibshirani & Friedman, ESL 3.4.1.
    ponytail: in-sample residuals slightly understate the spread (the fit absorbs some noise);
    the calibration check in scripts/m6_location.py shows whether it matters.
    """

    def __init__(self, alpha: float = 10.0, features: list[str] = FEATURES):
        self.alpha, self.features = alpha, list(features)

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return self.enc.transform(df[self.features]) if self.features else np.zeros((len(df), 0))

    def fit(self, train: pd.DataFrame) -> "LocationModel":
        """train needs columns x_ft, z_ft and the context features."""
        y = train[["x_ft", "z_ft"]].to_numpy()
        if self.features:
            self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[self.features])
            self.ridge = Ridge(alpha=self.alpha).fit(self._X(train), y)
            self.mu = self.ridge.predict(self._X(train))
        else:
            self.mu = np.tile(y.mean(axis=0), (len(y), 1))
        self.cov = np.cov((y - self.mu).T, ddof=1)
        self._y_mean = y.mean(axis=0)
        return self

    def mean(self, df: pd.DataFrame) -> np.ndarray:
        """(len(df), 2) predicted mean location in feet."""
        if not self.features:
            return np.tile(self._y_mean, (len(df), 1))
        return self.ridge.predict(self._X(df))

    def logpdf(self, df: pd.DataFrame) -> np.ndarray:
        """Log density (nats, per ft^2) of each row's true (x_ft, z_ft)."""
        return multivariate_normal(mean=np.zeros(2), cov=self.cov).logpdf(
            df[["x_ft", "z_ft"]].to_numpy() - self.mean(df))

    def inside(self, df: pd.DataFrame, level: float) -> np.ndarray:
        """True where the pitch is inside the model's central `level` ellipse (calibration check)."""
        r = df[["x_ft", "z_ft"]].to_numpy() - self.mean(df)
        d2 = np.einsum("ij,jk,ik->i", r, np.linalg.inv(self.cov), r)   # squared Mahalanobis distance
        return d2 <= chi2.ppf(level, df=2)

    def sample(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """(len(df), 2) locations DRAWN from the predicted distribution (never just the mean)."""
        return self.mean(df) + rng.multivariate_normal(np.zeros(2), self.cov, size=len(df))
