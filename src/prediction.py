"""Validated scenario prediction and unit conversion helpers."""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_PATH, DISTRICT, FEATURES, RAINFALL

PREDICTION_RESULT_FIELDS = {
    "district", "rainfall_mm", "predicted_yield_kg_ha", "predicted_yield_tonnes_ha",
    "productivity_category", "historical_reference_yield_kg_ha",
    "difference_from_reference_kg_ha", "percentage_difference_from_reference",
    "estimated_production_tonnes", "rainfall_outside_training_range", "extrapolation_warning",
}


def load_model_artifact(path=ARTIFACT_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}. Train it with `.venv/bin/python -m src.train_models` (or `python -m src.train_models`).")
    artifact = joblib.load(path)
    required = {"selected_model", "selected_model_name", "districts", "rainfall_min_mm", "rainfall_max_mm", "district_references_kg_ha", "category_thresholds_kg_ha", "final_training_n"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Model artifact is incomplete or incompatible; retrain with `python -m src.train_models`.")
    return artifact


def production_tonnes(yield_kg_ha: float, cultivated_area_ha: float) -> float:
    if not math.isfinite(float(yield_kg_ha)) or float(yield_kg_ha) < 0:
        raise ValueError("Yield must be a finite non-negative kg/ha value.")
    if not math.isfinite(float(cultivated_area_ha)) or float(cultivated_area_ha) <= 0:
        raise ValueError("Cultivated area must be a finite positive number of hectares.")
    return float(yield_kg_ha) * float(cultivated_area_ha) / 1000.0


def productivity_category(yield_kg_ha: float, thresholds: dict) -> str:
    """Classify relative to historical yield quartiles, not crop success."""
    value = float(yield_kg_ha)
    if value < float(thresholds["q25_kg_ha"]):
        return "POOR"
    if value < float(thresholds["q75_kg_ha"]):
        return "MODERATE"
    return "GOOD"


def normalize_prediction_result(result: dict, thresholds: dict) -> dict:
    """Validate the public prediction schema and upgrade prior session results.

    Older Streamlit sessions may retain a scenario created before categories
    were added. Derive only that missing category from saved training quartiles.
    """
    if not isinstance(result, dict):
        raise ValueError("Saved prediction result is invalid. Submit the scenario again.")
    normalized = dict(result)
    try:
        normalized.setdefault("productivity_category", productivity_category(normalized["predicted_yield_kg_ha"], thresholds))
        normalized.setdefault("extrapolation_warning", next((w for w in normalized.get("warnings", []) if "outside the training range" in w), None))
        normalized.setdefault("historical_reference_yield_kg_ha", None)
        normalized.setdefault("difference_from_reference_kg_ha", None)
        normalized.setdefault("percentage_difference_from_reference", None)
        normalized.setdefault("estimated_production_tonnes", None)
        normalized.setdefault("rainfall_outside_training_range", normalized["extrapolation_warning"] is not None)
        missing = PREDICTION_RESULT_FIELDS.difference(normalized)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Saved prediction result is incomplete. Submit the scenario again.") from exc
    if missing:
        raise ValueError(f"Saved prediction result is missing fields: {', '.join(sorted(missing))}. Submit the scenario again.")
    return normalized


def predict_scenario(artifact: dict, district: str, rainfall_mm: float, area_ha: float | None = None) -> dict:
    if district is None or not str(district).strip():
        raise ValueError("Select a district.")
    district = str(district).strip()
    if district not in artifact["districts"]:
        raise ValueError(f"District '{district}' was not represented in model training. Retrain with an appropriate district dataset before predicting for it.")
    try:
        rainfall_mm = float(rainfall_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("Rainfall must be a numeric value in millimetres.") from exc
    if not math.isfinite(rainfall_mm) or rainfall_mm < 0:
        raise ValueError("Rainfall must be a finite non-negative value in millimetres.")
    if area_ha is not None:
        try:
            area_ha = float(area_ha)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cultivated area must be a numeric value in hectares.") from exc
        if not math.isfinite(area_ha) or area_ha <= 0:
            raise ValueError("Cultivated area must be a finite positive value in hectares.")
    X = pd.DataFrame([{RAINFALL: rainfall_mm, DISTRICT: district}], columns=FEATURES)
    raw = float(np.asarray(artifact["selected_model"].predict(X)).reshape(-1)[0])
    if not math.isfinite(raw):
        raise ValueError("The model returned an invalid prediction; retrain and inspect the model artifact.")
    prediction = max(0.0, raw)
    clipped = raw < 0
    ref = artifact["district_references_kg_ha"].get(district)
    delta = prediction - float(ref) if ref is not None else None
    pct = (delta / float(ref) * 100) if delta is not None and float(ref) != 0 else None
    outside = rainfall_mm < artifact["rainfall_min_mm"] or rainfall_mm > artifact["rainfall_max_mm"]
    warnings = []
    if outside:
        warnings.append(f"Rainfall is outside the training range ({artifact['rainfall_min_mm']:.1f}–{artifact['rainfall_max_mm']:.1f} mm); this prediction extrapolates beyond observed data.")
    if clipped:
        warnings.append(f"The unconstrained model predicted {raw:.1f} kg/ha. The displayed estimate is clipped to 0 kg/ha; this signals model failure for this scenario.")
    if ref is None:
        warnings.append("No historical district reference is available.")
    result = {
        "district": district,
        "rainfall_mm": rainfall_mm,
        "predicted_yield_raw_kg_ha": raw,
        "predicted_yield_kg_ha": prediction,
        "predicted_yield_tonnes_ha": prediction / 1000.0,
        "estimated_production_tonnes": production_tonnes(prediction, area_ha) if area_ha is not None else None,
        "area_ha": area_ha,
        "historical_reference_yield_kg_ha": float(ref) if ref is not None else None,
        "difference_from_reference_kg_ha": delta,
        "percentage_difference_from_reference": pct,
        "rainfall_outside_training_range": outside,
        "extrapolation_warning": next((warning for warning in warnings if "outside the training range" in warning), None),
        "prediction_clipped": clipped,
        "productivity_category": productivity_category(prediction, artifact["category_thresholds_kg_ha"]),
        "warnings": warnings,
        "prediction_type": "Scenario estimate conditional on user-entered rainfall; this is not a future rainfall or yield forecast.",
    }
    return normalize_prediction_result(result, artifact["category_thresholds_kg_ha"])
