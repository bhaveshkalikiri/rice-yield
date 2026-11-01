import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.config import DATA_PATH, DISTRICT, FEATURES, RAINFALL, TARGET
from src.data_loader import load_dataset, validate_dataset
from src.evaluation import evaluate_candidates
from src.prediction import load_model_artifact, normalize_prediction_result, predict_scenario, productivity_category, production_tonnes
from src.train_models import DistrictHistoricalMeanRegressor, make_estimators, train_and_save


@pytest.fixture(scope="module")
def real_data():
    return load_dataset(DATA_PATH)


def simple_artifact(real_data):
    model = DistrictHistoricalMeanRegressor().fit(real_data[FEATURES], real_data[TARGET])
    return {
        "selected_model_name": "District historical mean baseline",
        "selected_model": model,
        "models": {"District historical mean baseline": model},
        "districts": sorted(real_data[DISTRICT].unique().tolist()),
        "district_references_kg_ha": real_data.groupby(DISTRICT)[TARGET].mean().to_dict(),
        "rainfall_min_mm": float(real_data[RAINFALL].min()),
        "rainfall_max_mm": float(real_data[RAINFALL].max()),
        "category_thresholds_kg_ha": {"q25_kg_ha": float(real_data[TARGET].quantile(.25)), "q75_kg_ha": float(real_data[TARGET].quantile(.75))},
        "final_training_n": len(real_data),
    }


def test_real_processed_dataset_loads_and_validates(real_data):
    assert real_data.shape[0] == 117
    assert {DISTRICT, RAINFALL, TARGET, "year"}.issubset(real_data.columns)
    assert real_data.duplicated([DISTRICT, "year"]).sum() == 0


def test_required_feature_validation_reports_missing_column(real_data):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(real_data.drop(columns=[RAINFALL]))


def test_invalid_negative_rainfall_in_dataset_is_rejected(real_data):
    invalid = real_data.copy()
    invalid.loc[invalid.index[0], RAINFALL] = -1
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_dataset(invalid)


def test_training_estimator_predicts_from_real_rows(real_data):
    estimator = DistrictHistoricalMeanRegressor().fit(real_data[FEATURES], real_data[TARGET])
    predictions = estimator.predict(real_data.iloc[:4][FEATURES])
    assert len(predictions) == 4
    assert np.isfinite(predictions).all()


def test_evaluation_is_reproducible_on_real_data(real_data):
    candidates = make_estimators()
    candidates = {"District historical mean baseline": candidates["District historical mean baseline"], "Linear Regression": candidates["Linear Regression"]}
    first, _, first_preds, (train, valid) = evaluate_candidates(real_data, candidates)
    second, _, second_preds, _ = evaluate_candidates(real_data, candidates)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_preds, second_preds)
    assert sorted(valid.year.unique()) == [2020, 2021]
    assert len(train) == 91 and len(valid) == 26
    assert train.year.max() < valid.year.min()
    assert first.validation_n.eq(26).all()
    assert first.district_grouped_validation_n_oof.eq(117).all()


def test_scenario_prediction_has_correct_units_and_optional_area(real_data):
    artifact = simple_artifact(real_data)
    district = artifact["districts"][0]
    result = predict_scenario(artifact, district, 600)
    assert result["predicted_yield_tonnes_ha"] == pytest.approx(result["predicted_yield_kg_ha"] / 1000)
    assert result["estimated_production_tonnes"] is None
    assert result["productivity_category"] in {"GOOD", "MODERATE", "POOR"}
    assert result["extrapolation_warning"] is None
    assert result["historical_reference_yield_kg_ha"] is not None
    assert result["difference_from_reference_kg_ha"] is not None
    with_area = predict_scenario(artifact, district, 600, 12.5)
    assert with_area["estimated_production_tonnes"] == pytest.approx(with_area["predicted_yield_kg_ha"] * 12.5 / 1000)


@pytest.mark.parametrize("rainfall", [-1, np.nan, np.inf, "not-a-number"])
def test_invalid_rainfall_scenario_is_rejected(real_data, rainfall):
    artifact = simple_artifact(real_data)
    with pytest.raises(ValueError, match="Rainfall"):
        predict_scenario(artifact, artifact["districts"][0], rainfall)


def test_missing_and_unknown_district_are_rejected(real_data):
    artifact = simple_artifact(real_data)
    with pytest.raises(ValueError, match="Select a district"):
        predict_scenario(artifact, " ", 600)
    with pytest.raises(ValueError, match="not represented"):
        predict_scenario(artifact, "UNSEEN DISTRICT", 600)


def test_rainfall_extrapolation_and_area_validation(real_data):
    artifact = simple_artifact(real_data)
    district = artifact["districts"][0]
    outside = predict_scenario(artifact, district, artifact["rainfall_max_mm"] + 100)
    assert outside["rainfall_outside_training_range"]
    assert any("outside the training range" in warning for warning in outside["warnings"])
    with pytest.raises(ValueError, match="positive"):
        predict_scenario(artifact, district, 600, 0)


def test_nonnegative_clipping_handles_invalid_negative_model_prediction(real_data):
    class NegativeModel:
        def predict(self, X):
            return np.full(len(X), -150.0)

    artifact = simple_artifact(real_data)
    artifact["selected_model"] = NegativeModel()
    result = predict_scenario(artifact, artifact["districts"][0], 600)
    assert result["predicted_yield_raw_kg_ha"] == -150
    assert result["predicted_yield_kg_ha"] == 0
    assert result["prediction_clipped"]
    assert any("model failure" in warning for warning in result["warnings"])


def test_model_artifact_load_and_missing_artifact_error(tmp_path, real_data):
    artifact = simple_artifact(real_data)
    good = tmp_path / "model.joblib"
    joblib.dump(artifact, good)
    loaded = load_model_artifact(good)
    assert loaded["selected_model_name"] == artifact["selected_model_name"]
    with pytest.raises(FileNotFoundError, match="Train it"):
        load_model_artifact(tmp_path / "missing.joblib")
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_text("not a joblib artifact")
    with pytest.raises(Exception):
        load_model_artifact(corrupt)


def test_productivity_category_quartile_edges():
    thresholds = {"q25_kg_ha": 1000, "q75_kg_ha": 2500}
    assert productivity_category(999, thresholds) == "POOR"
    assert productivity_category(1000, thresholds) == "MODERATE"
    assert productivity_category(2499, thresholds) == "MODERATE"
    assert productivity_category(2500, thresholds) == "GOOD"


def test_legacy_prediction_missing_category_is_upgraded_without_keyerror(real_data):
    thresholds = {"q25_kg_ha": float(real_data[TARGET].quantile(.25)), "q75_kg_ha": float(real_data[TARGET].quantile(.75))}
    legacy = {"predicted_yield_kg_ha": 3000.0, "predicted_yield_tonnes_ha": 3.0, "district": "GUNTUR", "rainfall_mm": 700.0}
    normalized = normalize_prediction_result(legacy, thresholds)
    assert normalized["productivity_category"] == "MODERATE"
    assert normalized["extrapolation_warning"] is None


def test_streamlit_theme_config_is_light_and_uses_product_palette():
    import tomllib

    config = tomllib.loads((Path(__file__).resolve().parents[1] / ".streamlit/config.toml").read_text())
    assert config["theme"]["base"] == "light"
    assert config["theme"]["backgroundColor"] == "#FAF3E0"
    assert config["theme"]["primaryColor"] == "#243B64"


def test_stage_b_fit_uses_every_eligible_observation(tmp_path, real_data, monkeypatch):
    import src.train_models as training

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(training, "METRICS_PATH", results_dir / "metrics.csv")
    monkeypatch.setattr(training, "METADATA_PATH", results_dir / "metadata.json")
    monkeypatch.setattr(training, "ROOT", tmp_path)
    result = train_and_save(artifact_path=tmp_path / "trained.joblib")
    assert result["artifact"]["final_training_n"] == len(real_data)
    assert result["metadata"]["final_model_fit_count"] == len(real_data)
    assert result["artifact"]["category_thresholds_kg_ha"]["source_n"] == len(real_data)
    assert {"District historical mean baseline", "Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"}.issubset(set(result["metrics"].model))
    assert result["artifact"]["selected_model_name"] == result["metrics"].sort_values("mae_kg_ha").iloc[0].model


def test_production_conversion_rejects_invalid_values():
    assert production_tonnes(2500, 40) == 100
    with pytest.raises(ValueError):
        production_tonnes(-1, 40)
    with pytest.raises(ValueError):
        production_tonnes(2500, np.inf)


def test_streamlit_pages_and_prediction_flow_run():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    view_select = next(widget for widget in app.selectbox if widget.label == "View")
    assert view_select.value == "Prediction"
    for page in ["Prediction", "Model performance", "Explore historical data", "About the project"]:
        view_select.set_value(page).run()
        assert not app.exception, f"Streamlit page failed: {page}"
        view_select = next(widget for widget in app.selectbox if widget.label == "View")
    view_select.set_value("Prediction").run()
    assert len(app.number_input) == 2
    assert [widget.label for widget in app.number_input] == ["June–September monsoon rainfall (mm)", "Cultivated area (hectares)"]
    assert app.number_input[1].value is None
    assert len(app.checkbox) == 0
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["scenario_prediction"]
    assert result["predicted_yield_kg_ha"] >= 0
    assert result["estimated_production_tonnes"] is None
    assert result["productivity_category"] in {"GOOD", "MODERATE", "POOR"}
    assert any("Productivity category" in item.value for item in app.markdown)
    assert len(app.get("plotly_chart")) >= 3
    legacy = dict(result)
    legacy.pop("productivity_category")
    app.session_state["scenario_prediction"] = legacy
    app.run()
    assert not app.exception
    assert app.session_state["scenario_prediction"]["productivity_category"] in {"GOOD", "MODERATE", "POOR"}
