"""Leakage-aware temporal and district-grouped model evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.config import DISTRICT, RAINFALL, TARGET, YEAR


def metric_row(name: str, y_true, y_pred, method: str, train_n: int, validation_n: int) -> dict:
    y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))
    return {
        "model": name,
        "mae_kg_ha": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kg_ha": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "validation_method": method,
        "training_n": int(train_n),
        "validation_n": int(validation_n),
    }


def temporal_holdout(data: pd.DataFrame, first_validation_year: int | None = None):
    years = sorted(data[YEAR].astype(int).unique())
    if len(years) < 4:
        raise ValueError("At least four years are required for a temporal holdout.")
    first = first_validation_year if first_validation_year is not None else years[-2]
    train = data.loc[data[YEAR] < first].copy()
    valid = data.loc[data[YEAR] >= first].copy()
    if train.empty or valid.empty:
        raise ValueError("Temporal holdout produced an empty training or validation set.")
    if set(valid[DISTRICT]) - set(train[DISTRICT]):
        raise ValueError("Temporal holdout includes unseen districts; choose a valid known-district holdout.")
    return train, valid


def evaluate_candidates(data: pd.DataFrame, estimators: dict, n_splits: int = 5):
    """Return chronological holdout metrics, grouped CV metrics, and predictions."""
    train, valid = temporal_holdout(data)
    X_train, y_train = train[[RAINFALL, DISTRICT]], train[TARGET]
    X_valid, y_valid = valid[[RAINFALL, DISTRICT]], valid[TARGET]
    group_splits = min(n_splits, data[DISTRICT].nunique())
    if group_splits < 2:
        raise ValueError("District-grouped evaluation requires at least two districts.")
    splitter = GroupKFold(n_splits=group_splits)
    rows, fitted_holdout, prediction_records = [], {}, []
    for name, estimator in estimators.items():
        holdout_model = clone(estimator).fit(X_train, y_train)
        holdout_pred = np.maximum(0, holdout_model.predict(X_valid))
        fitted_holdout[name] = holdout_model
        row = metric_row(name, y_valid, holdout_pred, "Chronological holdout: latest 2 years; known districts", len(train), len(valid))
        group_true, group_pred = [], []
        for fold_train, fold_valid in splitter.split(data, data[TARGET], groups=data[DISTRICT]):
            fold_model = clone(estimator).fit(data.iloc[fold_train][[RAINFALL, DISTRICT]], data.iloc[fold_train][TARGET])
            pred = np.maximum(0, fold_model.predict(data.iloc[fold_valid][[RAINFALL, DISTRICT]]))
            group_true.extend(data.iloc[fold_valid][TARGET].tolist())
            group_pred.extend(pred.tolist())
        grouped = metric_row(name, group_true, group_pred, f"{group_splits}-fold GroupKFold by district; unseen districts", len(data) * (group_splits - 1) // group_splits, len(data))
        row.update({
            "district_grouped_mae_kg_ha": grouped["mae_kg_ha"],
            "district_grouped_rmse_kg_ha": grouped["rmse_kg_ha"],
            "district_grouped_r2": grouped["r2"],
            "district_grouped_method": grouped["validation_method"],
            "district_grouped_training_n_per_fold_approx": grouped["training_n"],
            "district_grouped_validation_n_oof": grouped["validation_n"],
        })
        rows.append(row)
        prediction_records.extend({"model": name, "district": d, "year": int(y), "observed_yield_kg_ha": float(obs), "predicted_yield_kg_ha": float(pred)} for d, y, obs, pred in zip(valid[DISTRICT], valid[YEAR], y_valid, holdout_pred))
    return pd.DataFrame(rows), fitted_holdout, pd.DataFrame(prediction_records), (train, valid)
