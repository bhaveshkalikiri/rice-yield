"""Simple real-data baselines used in model comparison and prediction."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.config import DISTRICT


class DistrictHistoricalMeanRegressor(RegressorMixin, BaseEstimator):
    """Leakage-safe district mean, with a training-fold global-mean fallback."""

    def fit(self, X: pd.DataFrame, y):
        self.global_mean_ = float(np.mean(y))
        self.district_means_ = pd.Series(np.asarray(y), index=X[DISTRICT].astype(str)).groupby(level=0).mean().to_dict()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame):
        return np.asarray([self.district_means_.get(str(d), self.global_mean_) for d in X[DISTRICT]], dtype=float)
