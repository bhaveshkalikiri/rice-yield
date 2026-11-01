"""Load and validate the real, already processed district-year dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATA_PATH, DISTRICT, RAINFALL, ROOT, TARGET, YEAR

REQUIRED_COLUMNS = {DISTRICT, RAINFALL, TARGET, YEAR}


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    out = data.copy()
    out[DISTRICT] = out[DISTRICT].astype("string").str.strip()
    for col in (RAINFALL, TARGET, YEAR):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required district, year, rainfall and yield values must be present and numeric.")
    if not np.isfinite(out[[RAINFALL, TARGET, YEAR]].to_numpy(dtype=float)).all():
        raise ValueError("Required numeric values must be finite.")
    if (out[RAINFALL] < 0).any():
        raise ValueError("Monsoon rainfall cannot be negative.")
    if (out[TARGET] < 0).any():
        raise ValueError("Annual rice productivity cannot be negative.")
    if out[DISTRICT].eq("").any():
        raise ValueError("District names cannot be empty.")
    if out.duplicated([DISTRICT, YEAR]).any():
        raise ValueError("Dataset must have unique district-year observations.")
    return out.sort_values([YEAR, DISTRICT]).reset_index(drop=True)


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Read the pipeline's real merged dataset; never synthesize missing rows."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Merged rice/rainfall data not found: {path}. Run the existing `python run_pipeline.py` first.")
    return validate_dataset(pd.read_csv(path))
