"""Shared project paths and machine-learning feature definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/analysis_dataset.csv"
MODEL_DIR = ROOT / "models"
ARTIFACT_PATH = MODEL_DIR / "riceyield_model.joblib"
METRICS_PATH = ROOT / "results/ml_model_comparison.csv"
METADATA_PATH = ROOT / "results/model_metadata.json"
TARGET = "yield_kg_ha"
DISTRICT = "district"
RAINFALL = "monsoon_rainfall_mm"
YEAR = "year"
FEATURES = [RAINFALL, DISTRICT]
SEED = 20261009
