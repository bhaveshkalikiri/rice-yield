"""Train real-data candidates and select a model on a chronological holdout."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import ARTIFACT_PATH, DATA_PATH, DISTRICT, FEATURES, METADATA_PATH, METRICS_PATH, RAINFALL, ROOT, SEED, TARGET
from src.baselines import DistrictHistoricalMeanRegressor
from src.data_loader import load_dataset
from src.evaluation import evaluate_candidates


def make_estimators() -> dict:
    preprocessing = ColumnTransformer(
        transformers=[
            ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [DISTRICT]),
            ("rainfall", "passthrough", [RAINFALL]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return {
        "District historical mean baseline": DistrictHistoricalMeanRegressor(),
        "Linear Regression": Pipeline([("preprocess", preprocessing), ("model", LinearRegression())]),
        "Ridge Regression": Pipeline([("preprocess", preprocessing), ("model", Ridge(alpha=10.0))]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessing),
            ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=5, max_features=1.0, random_state=SEED, n_jobs=1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocess", preprocessing),
            ("model", GradientBoostingRegressor(n_estimators=60, learning_rate=.03, max_depth=1, min_samples_leaf=5, loss="huber", random_state=SEED)),
        ]),
    }


def _package_versions() -> dict:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "streamlit", "plotly"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def train_and_save(data_path=DATA_PATH, artifact_path=ARTIFACT_PATH) -> dict:
    import joblib

    data = load_dataset(data_path)
    estimators = make_estimators()
    metrics, _, holdout_predictions, (train, valid) = evaluate_candidates(data, estimators)
    metrics = metrics.sort_values(["mae_kg_ha", "model"], kind="stable").reset_index(drop=True)
    selected_name = str(metrics.iloc[0]["model"])
    # Stage B: fit the selected production model only after Stage A selection,
    # using every eligible historical row. Stage A metrics remain untouched.
    selected_model = estimators[selected_name].fit(data[FEATURES], data[TARGET])
    fitted_models = {name: estimator.fit(data[FEATURES], data[TARGET]) for name, estimator in estimators.items()}
    district_references = data.groupby(DISTRICT)[TARGET].mean().to_dict()
    artifact = {
        "schema_version": 1,
        "selected_model_name": selected_name,
        "selected_model": selected_model,
        "models": fitted_models,
        "features": FEATURES,
        "target": TARGET,
        "districts": sorted(data[DISTRICT].astype(str).unique().tolist()),
        "district_references_kg_ha": {str(k): float(v) for k, v in district_references.items()},
        "global_reference_kg_ha": float(data[TARGET].mean()),
        "rainfall_min_mm": float(data[RAINFALL].min()),
        "rainfall_max_mm": float(data[RAINFALL].max()),
        "yield_min_kg_ha": float(data[TARGET].min()),
        "yield_max_kg_ha": float(data[TARGET].max()),
        "observation_count": int(len(data)),
        "final_training_n": int(len(data)),
        "category_thresholds_kg_ha": {
            "q25_kg_ha": float(data[TARGET].quantile(0.25)),
            "q75_kg_ha": float(data[TARGET].quantile(0.75)),
            "source_n": int(len(data)),
            "definition": "Empirical 25th and 75th percentiles of all eligible historical yield observations.",
        },
        "district_count": int(data[DISTRICT].nunique()),
        "year_min": int(data.year.min()),
        "year_max": int(data.year.max()),
        "unit_conversion": "predicted production tonnes = predicted yield kg/ha * area ha / 1000",
        "negative_prediction_policy": "Predictions are clipped to zero; model validation metrics apply the same clipping.",
        "prediction_intervals": "Not implemented; no unsupported interval is displayed.",
    }
    metrics_path = METRICS_PATH
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    holdout_predictions.to_csv(ROOT / "results/ml_holdout_predictions.csv", index=False)
    artifact_path = artifact_path if hasattr(artifact_path, "parent") else ROOT / str(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    temporal_years = sorted(valid.year.astype(int).unique().tolist())
    resolved_data_path = Path(data_path)
    if not resolved_data_path.is_absolute():
        resolved_data_path = ROOT / resolved_data_path
    try:
        recorded_data_path = str(resolved_data_path.resolve().relative_to(ROOT))
    except ValueError:
        recorded_data_path = Path(resolved_data_path).name
    metadata = {
        "data_path": recorded_data_path,
        "selected_model": selected_name,
        "workflow": {"stage_a": "Compare candidates on chronological holdout and grouped district CV; metrics are held-out only.", "stage_b": "Refit selected candidate on every eligible observation for scenario prediction."},
        "selection_rule": "Lowest MAE on the latest-two-year chronological holdout among candidate models; predictions clipped at zero for both evaluation and inference.",
        "temporal_holdout_years": temporal_years,
        "training_count": int(len(train)),
        "temporal_validation_count": int(len(valid)),
        "full_dataset_count": int(len(data)),
        "final_model_fit_count": int(len(data)),
        "category_thresholds_kg_ha": artifact["category_thresholds_kg_ha"],
        "district_grouped_cv": "5-fold GroupKFold by district; estimates transfer to unseen districts and is reported separately, not used to select the known-district scenario model.",
        "features": FEATURES,
        "excluded_features": {"year": "Not requested by the scenario UI and would imply future-year forecasting; this app predicts a user-entered rainfall scenario.", "area_ha": "Used only for optional yield-to-production conversion.", "yield_calculated_kg_ha": "Derived from actual production/area and excluded to prevent target leakage."},
        "target": TARGET,
        "random_seed": SEED,
        "model_artifact": str(artifact_path.relative_to(ROOT)) if artifact_path.is_relative_to(ROOT) else str(artifact_path),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "versions": _package_versions(),
        "training_data_years": [int(data.year.min()), int(data.year.max())],
        "nonnegative_prediction_policy": artifact["negative_prediction_policy"],
        "validation_limitations": "Only 26 observations in two holdout years; known districts overlap training. Grouped district CV tests unseen-district transfer separately. Small sample means metrics are uncertain and negative R² is retained as observed.",
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return {"artifact": artifact, "metrics": metrics, "metadata": metadata, "artifact_path": artifact_path}


def main() -> None:
    result = train_and_save()
    best = result["metrics"].iloc[0]
    print(f"Selected: {result['artifact']['selected_model_name']}")
    print(f"Chronological holdout MAE={best.mae_kg_ha:.2f}, RMSE={best.rmse_kg_ha:.2f}, R²={best.r2:.3f}")
    print(f"Saved model: {result['artifact_path']}")


if __name__ == "__main__":
    main()
