from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
import xarray as xr

from src.download_imd_rainfall import download_years

RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
TABLES = ROOT / "results/tables"
AUDIT = ROOT / "results/data_audit"
FIG = ROOT / "figures"
SEED = 20261008
ICAR_URL = "https://icar-iirr.org/index.php/en/component/content/article/39-iirr-databases/214-rice-area-production-and-productivity-database"
IMD_URL = "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html"
BOUNDARY_REPO = "yashveeeeeeer/india-geodata"
BOUNDARY_RELEASE = "https://github.com/yashveeeeeeer/india-geodata/releases/download/admin%2Fdistricts/census-2011.zip"


def discover_rice_file() -> Path:
    candidates = []
    for current, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".venv", ".git", ".cache", ".pytest_cache", "__pycache__"}]
        for name in files:
            if Path(name).suffix.lower() in {".csv", ".xls", ".xlsx"} and "data/raw" not in current.replace("\\", "/"):
                candidates.append(Path(current) / name)
    ranked = []
    for path in candidates:
        try:
            if path.suffix.lower() == ".csv":
                text = path.read_text(encoding="utf-8-sig", errors="ignore")[:12000].lower()
                score = sum(word in text for word in ("district", "production", "yield", "state", "year"))
            else:
                score = 0
            if score >= 4:
                ranked.append((score, path))
        except OSError:
            continue
    if not ranked:
        raise FileNotFoundError("No supplied rice dataset with State/District/Year/Production/Yield fields was found.")
    ranked.sort(key=lambda x: (x[0], x[1].name == "Rice_Data_2026-10-08 (1).csv"), reverse=True)
    return ranked[0][1]


def read_rice(path: Path) -> tuple[pd.DataFrame, dict]:
    if path.suffix.lower() != ".csv":
        raise ValueError("Excel rice input is not supported yet; provide the ICAR CSV dataset.")
    raw = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header_idx = next((i for i, line in enumerate(raw) if {x.strip().lower() for x in next(csv.reader([line]))} >= {"state", "district", "year"}), None)
    if header_idx is None:
        raise ValueError("Could not locate a CSV header containing State, District and Year.")
    frame = pd.read_csv(path, skiprows=header_idx)
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    cols = {norm(str(c)): c for c in frame.columns}
    required = {"state": "state", "district": "district", "year": "year", "area": "area_ha", "production": "production_tonnes", "yield": "yield_kg_ha"}
    selected = {}
    for token, dest in required.items():
        matches = [col for n, col in cols.items() if token in n]
        if not matches:
            if token in ("area", "production", "yield"):
                continue
            raise ValueError(f"Missing required rice field: {token}")
        selected[dest] = matches[0]
    out = pd.DataFrame()
    for dest, source in selected.items():
        out[dest] = frame[source] if dest in ("state", "district") else pd.to_numeric(frame[source].astype(str).str.replace(",", "", regex=False), errors="coerce")
    for col in ("area_ha", "production_tonnes", "yield_kg_ha"):
        if col not in out:
            out[col] = np.nan
    out["state"] = out.state.astype("string").str.strip().str.upper()
    out["district"] = out.district.astype("string").str.strip().str.upper()
    out["year"] = pd.to_numeric(out.year, errors="coerce")
    out["yield_calculated_kg_ha"] = np.where(out.area_ha > 0, out.production_tonnes / out.area_ha * 1000, np.nan)
    out["yield_difference_kg_ha"] = out.yield_kg_ha - out.yield_calculated_kg_ha
    summary = {
        "input_file": path.name, "source_header_line_1based": header_idx + 1,
        "input_rows": len(frame), "empty_rows_removed": int(frame.isna().all(axis=1).sum()),
        "duplicate_district_year_records": int(out.duplicated(["state", "district", "year"], keep=False).sum()),
        "invalid_year_rows": int((out.year.notna() & ((out.year < 1901) | (out.year > datetime.now().year))).sum()),
        "missing_by_field": out.isna().sum().to_dict(),
        "negative_numeric_cells": {c: int((out[c] < 0).sum()) for c in ["area_ha", "production_tonnes", "yield_kg_ha"]},
        "nonpositive_area_rows": int((out.area_ha <= 0).sum()),
        "nonpositive_production_rows": int((out.production_tonnes <= 0).sum()),
        "nonpositive_reported_yield_rows": int((out.yield_kg_ha <= 0).sum()),
        "yield_discrepancy_abs_gt_10pct": int(((out.yield_calculated_kg_ha > 0) & (out.yield_kg_ha.notna()) & (abs(out.yield_difference_kg_ha) / out.yield_calculated_kg_ha > .10)).sum()),
        "yield_outliers_iqr_rule": int(((out.yield_kg_ha < out.yield_kg_ha.quantile(.25) - 1.5 * (out.yield_kg_ha.quantile(.75) - out.yield_kg_ha.quantile(.25))) | (out.yield_kg_ha > out.yield_kg_ha.quantile(.75) + 1.5 * (out.yield_kg_ha.quantile(.75) - out.yield_kg_ha.quantile(.25)))).sum()),
        "source_season_classification": "The ICAR page describes district-level rice area, production and productivity, but does not state a season. The data are treated as annual/combined rice productivity, not Kharif-only.",
    }
    out = out.dropna(subset=["state", "district", "year", "yield_kg_ha", "area_ha", "production_tonnes"])
    out["year"] = out.year.astype(int)
    return out, summary


def select_stable_ap(rice: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ap = rice[rice.state.str.contains("ANDHRA", na=False)].copy()
    counts = ap.groupby("year").district.nunique()
    # The supplied series changes geography at Telangana's 2014 separation and again
    # with Andhra Pradesh's 2022 district reorganization. Select the longest contiguous
    # span with a stable district roster, rather than fabricate an unsupported crosswalk.
    rosters = {int(y): tuple(sorted(g.district.unique())) for y, g in ap.groupby("year")}
    spans = []
    years = sorted(rosters)
    for start in years:
        # Before the 2014 Telangana separation, rows under Andhra Pradesh refer to
        # the former combined state; exclude that geography from an AP-only study.
        if start < 2013:
            continue
        for end in years:
            if end < start:
                continue
            ys = list(range(start, end + 1))
            if all(y in rosters for y in ys) and len(set(rosters[y] for y in ys)) == 1:
                spans.append((len(ys), start, end))
    _, y0, y1 = max(spans)
    stable = ap[ap.year.between(y0, y1)].copy()
    # require complete district-year reporting for the chosen stable panel
    roster = set(stable.district.unique())
    complete_years = [y for y, g in stable.groupby("year") if set(g.district) == roster]
    stable = stable[stable.year.isin(complete_years)].copy()
    info = {"selection_rule": "longest contiguous years with invariant Andhra Pradesh rice district roster and complete district-year reporting", "analysis_year_start": int(stable.year.min()), "analysis_year_end": int(stable.year.max()), "district_count": int(stable.district.nunique()), "included_districts": sorted(stable.district.unique()), "ap_year_district_counts": {str(int(k)): int(v) for k, v in counts.items()}, "excluded_ap_rows": int(len(ap) - len(stable)), "excluded_years_boundary_change": sorted(set(map(int, ap.year.unique())) - set(map(int, stable.year.unique())))}
    if stable.year.nunique() < 5:
        raise ValueError(f"Stable AP geography has too few years for useful analysis: {stable.year.unique()}")
    # Aggregated years created from subdistrict rows: sum area and production, then
    # recalculate area-weighted yield. For years already at the stable district units,
    # source values remain exactly as supplied.
    stable["analysis_district"] = stable.district
    stable["yield_kg_ha"] = stable["yield_kg_ha"].astype(float)
    return stable, info


def obtain_boundaries(out_dir: Path) -> Path:
    """Download Census 2011 district polygons from a documented India-geodata release."""
    out_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # A normal ZIP release is preferred; use GitHub's API tree to locate a Census 2011
    # GeoJSON fallback if the release asset name changes.
    target = PROCESSED / "census_2011_districts.geojson"
    if target.exists() and target.stat().st_size > 10000:
        return target
    local_shapes = list(out_dir.glob("*2011*Dist.shp")) or list(out_dir.glob("*.shp"))
    if local_shapes:
        import geopandas as gpd
        gpd.read_file(local_shapes[0]).to_file(target, driver="GeoJSON")
        return target
    session = requests.Session()
    api = f"https://api.github.com/repos/{BOUNDARY_REPO}/git/trees/main?recursive=1"
    response = session.get(api, timeout=60)
    response.raise_for_status()
    tree = response.json().get("tree", [])
    candidates = [x["path"] for x in tree if x["path"].lower().endswith(".geojson") and "2011" in x["path"].lower() and "district" in x["path"].lower()]
    if candidates:
        # Releasing Census 2011 district boundaries as geojson is preferred to today’s
        # units; pick a direct file with district geometries.
        source = "https://raw.githubusercontent.com/" + BOUNDARY_REPO + "/main/" + quote(candidates[0], safe="/")
        r = session.get(source, timeout=120)
        r.raise_for_status()
        raw_boundary = out_dir / "census_2011_districts_source.geojson"
        raw_boundary.write_bytes(r.content)
        if raw_boundary.stat().st_size > 10000:
            target.write_bytes(raw_boundary.read_bytes())
            return target
    shp_candidates = [x["path"] for x in tree if x["path"].lower().endswith(".shp") and "2011" in x["path"].lower() and "district" in x["path"].lower()]
    if shp_candidates:
        import geopandas as gpd
        shp_path = shp_candidates[0]
        base = shp_path[:-4]
        local = out_dir / Path(shp_path).name
        for extension in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            rel = base + extension
            if any(x["path"] == rel for x in tree):
                r = session.get("https://raw.githubusercontent.com/" + BOUNDARY_REPO + "/main/" + quote(rel, safe="/"), timeout=120)
                r.raise_for_status()
                (out_dir / Path(rel).name).write_bytes(r.content)
        if local.exists():
            gpd.read_file(local).to_file(target, driver="GeoJSON")
            return target
    raise RuntimeError("Could not find a directly downloadable Census 2011 district GeoJSON in the documented boundary repository. Check the repository's boundary release structure.")


def inspect_netcdf(path: Path, required_year: int) -> tuple[str, str, str]:
    with xr.open_dataset(path) as ds:
        if not ds.data_vars:
            raise ValueError(f"No rainfall variable in {path}")
        rain = next((v for v in ds.data_vars if "rain" in v.lower() or "rf" == v.lower()), list(ds.data_vars)[0])
        time_name = next((n for n in ds.coords if "time" in n.lower() or "date" in n.lower()), None)
        lat_name = next((n for n in ds.coords if n.lower() in ("lat", "latitude")), None)
        lon_name = next((n for n in ds.coords if n.lower() in ("lon", "longitude")), None)
        if not (time_name and lat_name and lon_name):
            raise ValueError(f"NetCDF lacks time/latitude/longitude coordinates: {path}; found {list(ds.coords)}")
        dates = pd.to_datetime(ds[time_name].values)
        if len(dates) < 350 or set(pd.DatetimeIndex(dates).year) != {required_year}:
            raise ValueError(f"Unexpected daily time axis in {path}: {len(dates)} records / years {set(pd.DatetimeIndex(dates).year)}")
        lat, lon = ds[lat_name].values, ds[lon_name].values
        lat_step = float(np.nanmedian(np.abs(np.diff(np.sort(np.unique(lat))))))
        lon_step = float(np.nanmedian(np.abs(np.diff(np.sort(np.unique(lon))))))
        if not (.24 <= lat_step <= .26 and .24 <= lon_step <= .26):
            raise ValueError(f"Coordinate spacing is not approximately 0.25 degrees in {path}: lat={lat_step}, lon={lon_step}")
        attrs = ds[rain].attrs
        units = str(attrs.get("units", "unspecified"))
        if not (np.nanmin(lat) < 10 and np.nanmax(lat) > 30 and np.nanmin(lon) < 70 and np.nanmax(lon) > 90):
            raise ValueError(f"Unexpected spatial extent in {path}")
        return rain, time_name, units


def monsoon_mask(dates) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    return (dates.month >= 6) & (dates.month <= 9)


def validate_rainfall_year(dates, expected_year: int) -> None:
    years = set(pd.DatetimeIndex(pd.to_datetime(dates)).year)
    if years != {int(expected_year)}:
        raise ValueError(f"Rainfall file year mismatch: expected {expected_year}, found {sorted(years)}")


def aggregate_monsoon(daily_grid: np.ndarray, dates, cell_indices: np.ndarray, expected_year: int) -> float:
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    validate_rainfall_year(dates, expected_year)
    values = np.asarray(daily_grid)
    if values.ndim != 2 or values.shape[0] != len(dates):
        raise ValueError("daily_grid must have shape (time, grid_cell)")
    daily = np.nanmean(values[monsoon_mask(dates)][:, cell_indices], axis=1)
    return float(np.nansum(daily)) if np.isfinite(daily).any() else np.nan


def merge_district_year(rice: pd.DataFrame, rainfall: pd.DataFrame) -> pd.DataFrame:
    return rice.merge(rainfall, on=["district", "year"], how="inner", validate="one_to_one", indicator=True)


def quadratic_turning_point(beta1: float, beta2: float, rainfall_min: float, rainfall_max: float) -> tuple[float, bool]:
    if not np.isfinite(beta1) or not np.isfinite(beta2) or beta2 >= 0:
        return np.nan, False
    point = float(-beta1 / (2 * beta2) * 100)
    return point, bool(rainfall_min <= point <= rainfall_max)


def district_rainfall(files: list[Path], districts: list[str], boundary_path: Path) -> tuple[pd.DataFrame, dict]:
    import geopandas as gpd
    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        raise ValueError("Census district boundary file has no CRS")
    gdf = gdf.to_crs("EPSG:4326")
    # Accommodate the DataMeet / Census attribute naming without fuzzy name invention.
    name_col = next((c for c in gdf.columns if c.lower() in {"district", "district_n", "dist_name", "dt_nm", "name_2", "districtname", "distname"}), None)
    state_col = next((c for c in gdf.columns if c.lower() in {"state", "state_n", "st_name", "st_nm", "name_1", "statename"}), None)
    if not name_col:
        raise ValueError(f"District names not found in boundary attributes: {list(gdf.columns)}")
    canonical = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper().replace("&", "AND"))
    gdf["_district"] = gdf[name_col].map(canonical)
    if state_col:
        gdf["_state"] = gdf[state_col].map(canonical)
        gdf = gdf[gdf._state.str.contains("ANDHRAPRADESH", na=False)]
    aliases = {"SPSRNELLORE": ["NELLORE", "POTTISRIRAMULUNELLORE", "SRIPOTTISRIRAMULUNELLORE"], "YSR": ["KADAPA", "YSRKADAPA"], "VISAKHAPATANAM": ["VISAKHAPATNAM"]}
    wanted = {canonical(x): x for x in districts}
    for target_name, alternate_names in aliases.items():
        if target_name in wanted:
            for alternate in alternate_names:
                wanted[canonical(alternate)] = wanted[target_name]
    gdf = gdf[gdf._district.isin(wanted)]
    gdf["district"] = gdf._district.map(wanted)
    missing = sorted(set(districts) - set(gdf.district.unique()))
    if missing:
        raise ValueError(f"Could not match analysis districts to Census 2011 boundaries: {missing}; boundary names: {sorted(gdf._district.unique())}")
    # Point-in-polygon assigns each IMD grid cell by its center; each cell has equal
    # area at this resolution, so the within-district arithmetic mean is transparent.
    records = []
    for f in files:
        year = int(re.search(r"(19|20)\d{2}", f.name).group())
        var, tname, units = inspect_netcdf(f, year)
        with xr.open_dataset(f) as ds:
            rain = ds[var]
            coords = rain.coords
            lat_name = next(n for n in coords if n.lower() in ("lat", "latitude"))
            lon_name = next(n for n in coords if n.lower() in ("lon", "longitude"))
            time_name = tname
            lat_values, lon_values = ds[lat_name].values, ds[lon_name].values
            # Handle 1D or 2D spatial grids.
            if lat_values.ndim == lon_values.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
            else:
                lat_grid, lon_grid = lat_values, lon_values
            points = gpd.GeoDataFrame({"lat": lat_grid.ravel(), "lon": lon_grid.ravel()}, geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()), crs="EPSG:4326")
            sjoin = gpd.sjoin(points, gdf[["district", "geometry"]], how="inner", predicate="within")
            idx_by_district = {d: sjoin.index[sjoin.district == d].to_numpy(dtype=int) for d in districts}
            if lat_values.ndim == lon_values.ndim == 1:
                values = rain.transpose(time_name, lat_name, lon_name).values.reshape(len(ds[time_name]), -1)
            else:
                other_dims = [dim for dim in rain.dims if dim != time_name]
                values = rain.transpose(time_name, *other_dims).values.reshape(len(ds[time_name]), -1)
            dates = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values))
            validate_rainfall_year(dates, year)
            for district in districts:
                idx = idx_by_district[district]
                if len(idx) == 0:
                    continue
                # Missing cell values remain missing; do not interpolate/fill.
                total = aggregate_monsoon(values, dates, idx, year)
                records.append({"district": district, "year": year, "monsoon_rainfall_mm": total, "grid_cells": len(idx), "rain_units_source": units})
    result = pd.DataFrame(records)
    if result.duplicated(["district", "year"]).any():
        raise ValueError("Rainfall table has duplicate district-year rows")
    return result, {"boundary_source": "India Geodata repository, Census 2011 district boundaries (DataMeet source)", "boundary_url": "https://github.com/yashveeeeeeer/india-geodata/tree/main/data/administrative/districts/census-2011", "crs": "EPSG:4326", "rainfall_method": "Unweighted arithmetic mean of IMD 0.25-degree grid-cell centers within Census 2011 district polygon; daily district means summed over June–September.", "units": "mm (source units checked against IMD documentation; see file metadata)", "missing_cells": "No interpolation; valid daily grid-cell values averaged, and all-missing season totals remain missing."}


def bootstrap_turning_point(data: pd.DataFrame, draws: int = 1000) -> tuple[float, float, float, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    districts = data.district.unique()
    values = []
    for _ in range(draws):
        sampled = rng.choice(districts, size=len(districts), replace=True)
        pieces = []
        for i, d in enumerate(sampled):
            part = data.loc[data.district == d].copy()
            part["district"] = f"{d}_{i}"
            pieces.append(part)
        boot = pd.concat(pieces, ignore_index=True)
        try:
            b2, b1, _ = np.polyfit(boot.rainfall_100mm.to_numpy(), boot.yield_kg_ha.to_numpy(), 2)
            tp = -b1 / (2 * b2) * 100
            if b2 < 0 and np.isfinite(tp) and data.monsoon_rainfall_mm.min() <= tp <= data.monsoon_rainfall_mm.max():
                values.append(float(tp))
        except Exception:
            continue
    if values:
        return float(np.median(values)), float(np.quantile(values, .025)), float(np.quantile(values, .975)), pd.DataFrame({"turning_point_mm": values})
    return np.nan, np.nan, np.nan, pd.DataFrame({"turning_point_mm": []})


def save_figures(data: pd.DataFrame, linear, quadratic, fe, bootstrap: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for folder in (FIG / "eda", FIG / "models", FIG / "diagnostics"):
        folder.mkdir(parents=True, exist_ok=True)
    def save(fig, path):
        fig.text(.01, .005, "Source: ICAR-IIRR rice database and IMD gridded rainfall; district aggregation uses Census 2011 boundaries.", fontsize=7, color="#555555")
        fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(); ax.hist(data.yield_kg_ha, bins=18, color="#4472C4"); ax.set(title="Annual rice productivity distribution", xlabel="Annual rice productivity (kg/ha)", ylabel="District-years"); save(fig, FIG/"eda/01_yield_distribution.png")
    fig, ax = plt.subplots(); ax.hist(data.monsoon_rainfall_mm, bins=18, color="#70AD47"); ax.set(title="June–September rainfall distribution", xlabel="Monsoon rainfall (mm)", ylabel="District-years"); save(fig, FIG/"eda/02_rainfall_distribution.png")
    fig, ax = plt.subplots(); ax.scatter(data.monsoon_rainfall_mm,data.yield_kg_ha,alpha=.7); ax.set(title="Rainfall and annual rice productivity",xlabel="June–September rainfall (mm)",ylabel="Annual rice productivity (kg/ha)"); save(fig,FIG/"eda/03_rainfall_vs_yield.png")
    for field, title, ylabel, fname in [("monsoon_rainfall_mm","Yearly monsoon rainfall","Mean district rainfall (mm)","04_yearly_rainfall.png"),("yield_kg_ha","Yearly annual rice productivity","Mean district productivity (kg/ha)","05_yearly_yield.png")]:
        fig,ax=plt.subplots(); data.groupby("year")[field].mean().plot(ax=ax,marker="o"); ax.set(title=title,xlabel="Year",ylabel=ylabel); save(fig,FIG/"eda"/fname)
    data["rainfall_bin"] = pd.qcut(data.monsoon_rainfall_mm, q=min(5,data.monsoon_rainfall_mm.nunique()), duplicates="drop")
    bins=data.groupby("rainfall_bin",observed=True).agg(rainfall=("monsoon_rainfall_mm","mean"),yield_mean=("yield_kg_ha","mean"),yield_sd=("yield_kg_ha","sem")).reset_index()
    fig,ax=plt.subplots(); ax.errorbar(bins.rainfall,bins.yield_mean,yerr=1.96*bins.yield_sd,marker="o",capsize=3); ax.set(title="Mean productivity across rainfall quantiles",xlabel="Mean June–September rainfall (mm)",ylabel="Mean annual rice productivity (kg/ha)"); save(fig,FIG/"eda/06_rainfall_bins.png")
    x=np.linspace(data.monsoon_rainfall_mm.min(),data.monsoon_rainfall_mm.max(),250); scaled=x/100
    fig,ax=plt.subplots(); ax.scatter(data.monsoon_rainfall_mm,data.yield_kg_ha,alpha=.4); ax.plot(x,linear.predict(pd.DataFrame({"rainfall_100mm":scaled})),color="black"); ax.set(title="Linear association",xlabel="June–September rainfall (mm)",ylabel="Annual rice productivity (kg/ha)"); save(fig,FIG/"models/07_linear_fit.png")
    pred=pd.DataFrame({"rainfall_100mm":scaled}); fig,ax=plt.subplots(); ax.scatter(data.monsoon_rainfall_mm,data.yield_kg_ha,alpha=.35); ax.plot(x,quadratic.predict(pred),color="#C00000"); ax.set(title="Quadratic association",xlabel="June–September rainfall (mm)",ylabel="Annual rice productivity (kg/ha)"); save(fig,FIG/"models/08_quadratic_fit.png")
    fig,ax=plt.subplots();
    if len(bootstrap): ax.hist(bootstrap.turning_point_mm,bins=25,color="#ED7D31");
    ax.set(title="Cluster bootstrap turning-point estimates",xlabel="Turning point (mm)",ylabel="Bootstrap draws retained"); save(fig,FIG/"models/09_bootstrap_turning_point.png")
    fig,ax=plt.subplots(); centered=data.assign(yield_within=data.yield_kg_ha-data.groupby("district").yield_kg_ha.transform("mean"),rain_within=data.monsoon_rainfall_mm-data.groupby("district").monsoon_rainfall_mm.transform("mean")); ax.scatter(centered.rain_within,centered.yield_within,alpha=.6); ax.set(title="District-demeaned rainfall and productivity",xlabel="Within-district rainfall deviation (mm)",ylabel="Within-district productivity deviation (kg/ha)"); save(fig,FIG/"models/10_fixed_effects_relationship.png")
    fitted=quadratic.fittedvalues; resid=quadratic.resid
    fig,ax=plt.subplots(); ax.scatter(fitted,resid,alpha=.6); ax.axhline(0,color="black",lw=1); ax.set(title="Quadratic model residuals",xlabel="Fitted productivity (kg/ha)",ylabel="Residual (kg/ha)"); save(fig,FIG/"diagnostics/11_residual_fitted.png")
    from scipy import stats
    fig=plt.figure(); ax=fig.add_subplot(111); stats.probplot(resid,dist="norm",plot=ax); ax.set_title("Quadratic model normal Q–Q diagnostic"); save(fig,FIG/"diagnostics/12_qq_plot.png")
    influence=quadratic.get_influence(); fig,ax=plt.subplots(); ax.scatter(influence.hat_matrix_diag,influence.cooks_distance[0],alpha=.7); ax.set(title="Influence diagnostic",xlabel="Leverage",ylabel="Cook’s distance"); save(fig,FIG/"diagnostics/13_influence_diagnostics.png")


def run_analysis(data: pd.DataFrame, rice_meta: dict, geo_meta: dict) -> dict:
    TABLES.mkdir(parents=True, exist_ok=True); AUDIT.mkdir(parents=True, exist_ok=True)
    data = data.dropna(subset=["yield_kg_ha", "monsoon_rainfall_mm"]).copy()
    data["rainfall_100mm"] = data.monsoon_rainfall_mm / 100
    data.to_csv(PROCESSED / "analysis_dataset.csv", index=False)
    data.describe(include="all").transpose().to_csv(TABLES/"01_data_summary.csv")
    data.isna().sum().rename("missing_count").to_csv(TABLES/"02_missingness.csv")
    pd.DataFrame([{"input_file": rice_meta["input_file"], "input_rows": rice_meta["input_rows"], "empty_rows_removed": rice_meta["empty_rows_removed"], "invalid_year_rows": rice_meta["invalid_year_rows"], "duplicate_district_year_records": rice_meta["duplicate_district_year_records"], "yield_discrepancy_abs_gt_10pct": rice_meta["yield_discrepancy_abs_gt_10pct"], "yield_outliers_iqr_rule": rice_meta["yield_outliers_iqr_rule"], "nonpositive_area_rows": rice_meta["nonpositive_area_rows"], "nonpositive_production_rows": rice_meta["nonpositive_production_rows"], "nonpositive_reported_yield_rows": rice_meta["nonpositive_reported_yield_rows"], **{f"missing_{k}": v for k,v in rice_meta["missing_by_field"].items()}, **{f"negative_{k}": v for k,v in rice_meta["negative_numeric_cells"].items()}}]).to_csv(TABLES/"03_rice_cleaning_summary.csv",index=False)
    data.groupby("year").monsoon_rainfall_mm.agg(["count","mean","std","min","max"]).to_csv(TABLES/"04_rainfall_summary.csv")
    pd.DataFrame([{"merged_observations":len(data),"districts":data.district.nunique(),"years":data.year.nunique(),"min_year":data.year.min(),"max_year":data.year.max()}]).to_csv(TABLES/"05_merge_summary.csv",index=False)
    linear=smf.ols("yield_kg_ha ~ rainfall_100mm",data=data).fit(cov_type="HC3")
    quadratic=smf.ols("yield_kg_ha ~ rainfall_100mm + I(rainfall_100mm ** 2)",data=data).fit(cov_type="HC3")
    fe=smf.ols("yield_kg_ha ~ rainfall_100mm + I(rainfall_100mm ** 2) + C(district) + C(year)",data=data).fit(cov_type="cluster",cov_kwds={"groups":data.district})
    def coef_table(model):
        return pd.DataFrame({"term":model.params.index,"estimate":model.params.values,"std_error":model.bse.values,"p_value":model.pvalues.values,"ci_low":model.conf_int()[0].values,"ci_high":model.conf_int()[1].values})
    coef_table(linear).to_csv(TABLES/"06_linear_model.csv",index=False); coef_table(quadratic).to_csv(TABLES/"07_quadratic_model.csv",index=False); coef_table(fe).to_csv(TABLES/"11_fixed_effects.csv",index=False)
    pd.DataFrame([{"model":"linear","n":int(linear.nobs),"r_squared":linear.rsquared,"aic":linear.aic},{"model":"quadratic","n":int(quadratic.nobs),"r_squared":quadratic.rsquared,"aic":quadratic.aic},{"model":"district_and_year_fixed_effects_quadratic","n":int(fe.nobs),"r_squared":fe.rsquared,"aic":fe.aic}]).to_csv(TABLES/"08_model_comparison.csv",index=False)
    b1=quadratic.params.get("rainfall_100mm",np.nan); b2=quadratic.params.get("I(rainfall_100mm ** 2)",np.nan)
    tp, inside=quadratic_turning_point(b1,b2,data.monsoon_rainfall_mm.min(),data.monsoon_rainfall_mm.max())
    pd.DataFrame([{"beta2":b2,"beta2_p_value":quadratic.pvalues.get("I(rainfall_100mm ** 2)",np.nan),"turning_point_mm":tp,"inside_observed_range":inside,"rainfall_min_mm":data.monsoon_rainfall_mm.min(),"rainfall_max_mm":data.monsoon_rainfall_mm.max()}]).to_csv(TABLES/"09_turning_point.csv",index=False)
    bt, lo, hi, boot=bootstrap_turning_point(data)
    boot.to_csv(TABLES/"10_bootstrap_turning_point.csv",index=False)
    # Robustness: district-clustered SEs; log yield (positive values); exclude the
    # single most extreme rainfall observation; within-period quadratic estimates.
    robustness=[]
    for label, subset in [("full_sample",data),("exclude_rainfall_extremes",data[data.monsoon_rainfall_mm.between(data.monsoon_rainfall_mm.quantile(.01),data.monsoon_rainfall_mm.quantile(.99))])]:
        m=smf.ols("yield_kg_ha ~ rainfall_100mm + I(rainfall_100mm ** 2)",data=subset).fit(cov_type="cluster",cov_kwds={"groups":subset.district})
        robustness.append({"specification":label,"n":len(subset),"beta2":m.params.iloc[2],"beta2_p_value":m.pvalues.iloc[2],"r_squared":m.rsquared})
    logged=data[data.yield_kg_ha>0].copy(); logged["log_yield"]=np.log(logged.yield_kg_ha)
    lm=smf.ols("log_yield ~ rainfall_100mm + I(rainfall_100mm ** 2)",data=logged).fit(cov_type="HC3")
    robustness.append({"specification":"log_yield_hc3","n":len(logged),"beta2":lm.params.iloc[2],"beta2_p_value":lm.pvalues.iloc[2],"r_squared":lm.rsquared})
    pd.DataFrame(robustness).to_csv(TABLES/"12_robustness.csv",index=False)
    infl=quadratic.get_influence(); pd.DataFrame({"district":data.district,"year":data.year,"leverage":infl.hat_matrix_diag,"cooks_distance":infl.cooks_distance[0],"studentized_residual":infl.resid_studentized_external}).to_csv(TABLES/"13_influence_diagnostics.csv",index=False)
    save_figures(data,linear,quadratic,fe,boot)
    robust_direction = all(row["beta2"] < 0 for row in robustness)
    robust_significance = all(row["beta2_p_value"] < .05 for row in robustness)
    return {"data":data,"linear":linear,"quadratic":quadratic,"fe":fe,"turning_point":tp,"turning_point_inside":inside,"bootstrap_median":bt,"bootstrap_low":lo,"bootstrap_high":hi,"bootstrap_valid_draws":len(boot),"geo_meta":geo_meta,"rice_meta":rice_meta,"robustness":robustness,"robust_direction":robust_direction,"robust_significance":robust_significance}


def write_report(result: dict, geo_info: dict, selection: dict) -> None:
    d=result["data"]; linear=result["linear"]; quad=result["quadratic"]; fe=result["fe"]
    b2=quad.params.iloc[2]; p2=quad.pvalues.iloc[2]; beta=quad.params.iloc[1]
    rmin,rmax=d.monsoon_rainfall_mm.min(),d.monsoon_rainfall_mm.max()
    y_min,y_max=d.yield_kg_ha.min(),d.yield_kg_ha.max()
    def f(x,n=2): return "not estimable" if pd.isna(x) else f"{x:,.{n}f}"
    def pvalue(x): return "p < 0.001" if x < .001 else f"p = {x:.3f}"
    linear_text=f"The pooled linear coefficient was {f(linear.params.iloc[1]*100)} kg/ha per 100 mm ({pvalue(linear.pvalues.iloc[1])}; HC3 robust SE)."
    quad_text=f"The quadratic coefficient was {f(b2)} ({pvalue(p2)}; HC3 robust SE), with a linear coefficient of {f(beta)} per 100 mm."
    tp_text=f"The algebraic turning point was {f(result['turning_point'])} mm; it was {'inside' if result['turning_point_inside'] else 'outside or undefined within'} the observed range ({f(rmin)}–{f(rmax)} mm)." if np.isfinite(result['turning_point']) else "No maximum turning point was estimated because the fitted quadratic did not have a negative quadratic coefficient."
    conclusion="The fitted pooled quadratic is consistent with an inverted-U pattern" if b2 < 0 and p2 < .05 and result["turning_point_inside"] else "The analysis does not establish an inverted-U relationship with an interior rainfall maximum"
    text=f"""# Research report: Estimating the Monsoon Rainfall Threshold for Maximum Rice Productivity

## Abstract

This district-year study examines whether June–September monsoon rainfall is associated with annual rice productivity in Andhra Pradesh. ICAR-IIRR district data are matched to official IMD 0.25° daily gridded rainfall using Census 2011 district boundaries. The analysis uses {len(d)} matched district-years across {d.district.nunique()} stable district units from {d.year.min()} to {d.year.max()}. {conclusion}. All results are associational and do not identify causal effects.

## Introduction

Rice productivity depends on water availability, but both water stress and excess rainfall may constrain crop performance. A quadratic model provides a direct descriptive test of whether the relationship bends downward over the observed rainfall range. A fitted curve alone does not show that rainfall causes productivity changes; irrigation, temperature, soils, crop technology and other factors are not fully controlled.

## Research question

Is there an inverted-U relationship between June–September monsoon rainfall and annual rice productivity across Andhra Pradesh districts, and what rainfall level is associated with maximum predicted productivity?

## Hypothesis

The working hypothesis is that the quadratic rainfall coefficient is negative and that the implied maximum lies within the observed rainfall support. The estimates, not the hypothesis, determine the conclusion.

## Background

Rainfall is an imperfect proxy for crop water availability. Irrigation can buffer rainfall deficits, while rainfall timing, intensity and flooding matter in ways that seasonal totals cannot capture. The analysis therefore treats rainfall as a district-level seasonal exposure measure rather than a complete measure of crop water conditions.

## Data sources

The rice input is the user-supplied ICAR-IIRR district-level Rice Area, Production and Productivity Database, described by ICAR-IIRR as covering 1997–2023. The public description does not specify whether the figures are season-specific; this report calls the outcome annual rice productivity and does not label it Kharif-only. IMD daily rainfall comes from the official 0.25° gridded NetCDF archive. District polygons are Census 2011 boundaries obtained through the India Geodata repository, whose district README documents DataMeet Census boundary files.

## Study area

Rice records change district geography over time: the combined Andhra Pradesh roster appears through 2012, the 13-district residual state appears in 2013–2021, and 26 districts appear from 2022. To avoid combining the pre-bifurcation state or imposing an unsupported crosswalk on the 2022 reorganization, the main analysis selects the longest complete contiguous period with an invariant Andhra Pradesh roster beginning in 2013. The {d.district.nunique()} units are {', '.join(sorted(d.district.unique()))}

## Variables

The outcome is source-reported annual rice yield in kg/ha. The exposure is the arithmetic mean across IMD grid-cell centers inside each 2011 district polygon for each day, summed over June–September in mm. In the regression, rainfall is scaled in 100 mm units to improve coefficient readability. Source yield is retained; a separate production/area-derived yield check is written to the cleaning outputs.

## Data preparation

The CSV parser detects the actual header after a leading metadata line, standardizes fields, converts numeric columns, checks missing, duplicate and negative values, and calculates production/area × 1,000. Rice yield is not overwritten by the calculation. District-year observations outside the selected stable geography are excluded and counted in the audit. No missing yield or rainfall value is imputed.

## Rainfall processing

Only years overlapping the stable rice analysis period are requested from IMD. Each file is checked for NetCDF signature, daily timestamps, year, coordinate names, spatial extent and grid spacing. Rainfall is spatially summarized with equal weights for grid-cell centers within district polygons. Daily missing values are excluded from that day's available-cell mean; an all-missing season remains missing.

## Statistical methodology

The pooled linear model regresses yield on rainfall. The quadratic model adds squared rainfall. HC3 heteroskedasticity-robust standard errors are reported for pooled models. The implied stationary point is calculated as −β₁/(2β₂), converted back to mm, and treated as a maximum only when β₂ is negative and the point falls inside the observed range. A 1,000-draw district-cluster bootstrap resamples districts with replacement and retains only negative-curvature roots within the observed rainfall range. A district and year fixed-effects quadratic model uses district-clustered standard errors. Robustness checks include exclusion of the outer 1% rainfall observations and log yield.

## Exploratory analysis

The analysis includes distribution plots, rainfall-yield scatterplots, yearly means and rainfall-quantile summaries. The observed rainfall range was {f(rmin)}–{f(rmax)} mm; annual yield ranged from {f(y_min)} to {f(y_max)} kg/ha.

## Linear regression

{linear_text} The coefficient summarizes the pooled association and is not a causal effect.

## Quadratic regression

{quad_text} {"The negative curvature is statistically distinguishable from zero at the 5% level." if b2<0 and p2<.05 else "The estimates do not provide conventional 5% evidence of negative curvature."}

## Turning-point estimation

{tp_text} This point is meaningful as an observed-range maximum only if the fitted curvature is negative and the root lies in the data support.

## Bootstrap uncertainty

Among {result['bootstrap_valid_draws']} valid cluster-bootstrap draws, the retained turning-point median was {f(result['bootstrap_median'])} mm (95% percentile interval {f(result['bootstrap_low'])}–{f(result['bootstrap_high'])} mm). Because draws not producing an interior maximum are not retained, this interval is conditional on such a maximum being estimated and must be read alongside the retained draw count.

## Fixed-effects analysis

The district and year fixed-effects quadratic coefficient on squared rainfall was {f(fe.params.iloc[2])} (clustered {pvalue(fe.pvalues.iloc[2])}). This {"supports" if fe.params.iloc[2]<0 and fe.pvalues.iloc[2]<.05 else "does not provide statistically significant negative-curvature support for"} the pooled inverted-U conclusion. It controls for time-invariant district differences and common year effects, but not time-varying confounders.

## Robustness checks

Across the prespecified robustness estimates, negative curvature was {"directionally consistent" if result['robust_direction'] else "not directionally consistent"}; statistical significance at 5% was {"consistent" if result['robust_significance'] else "not consistent across all checks"}. The table reports estimates after trimming the outer 1% rainfall observations and using log productivity. Robustness estimates are descriptive and their uncertainty can be large in a short panel.

## Results

There were {len(d)} matched observations from {d.district.nunique()} districts over {d.year.min()}–{d.year.max()}. {conclusion}. {linear_text} {quad_text} {tp_text}

## Discussion

Rainfall can be associated with annual rice productivity through several channels, but this data design cannot distinguish rainfall effects from irrigation access, temperature, soil, pests, crop varieties, fertilizer, technology or changing management. District-level rainfall averages also introduce measurement error and conceal within-district variability and rainfall timing.

## Limitations

The ICAR public description does not clarify crop-season coverage. The study uses the source's annual/combined productivity label. Administrative changes require restricting the main analysis to a stable roster, shortening the time series and excluding unmatched years. Census 2011 boundaries are a proxy for within-district spatial assignment across the panel. IMD 0.25° rainfall may smooth local extremes; grid-cell center inclusion is an approximate spatial aggregation. The observational design does not establish causality, and the short panel limits fixed-effects precision.

## Conclusion

The data {('support' if b2<0 and p2<.05 and result['turning_point_inside'] else 'do not establish')} the proposed inverted-U rainfall–productivity association over the observed sample. No recommendation about an agronomic optimum should be made from this district-level association alone.

## Reproducibility

Run `python run_pipeline.py` from the repository root. The pipeline discovers the supplied rice CSV, downloads the relevant official IMD files and district boundaries, then recreates processed data, tables, figures, audit documents and this report. Random seed: {SEED}.

## References

- ICAR-Indian Institute of Rice Research. Rice Area, Production and Productivity Database. {ICAR_URL}
- India Meteorological Department. Daily Gridded Rainfall (0.25° × 0.25°), NetCDF archive. {IMD_URL}
- Pai, D. S., Sridhar, L., Rajeevan, M., Sreejith, O. P., Satbhai, N. S., & Mukhopadhyay, B. (2014). Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region. *MAUSAM, 65*(1), 1–18.
- India Geodata. District administrative boundary repository (Census 2011; source: DataMeet Maps). https://github.com/yashveeeeeeer/india-geodata/tree/main/data/administrative/districts/census-2011
"""
    report=ROOT/"report/research_report.md"; report.parent.mkdir(parents=True,exist_ok=True); report.write_text(text)
    result_text=f"""# Final results

- Matched observations: {len(d)}
- Districts: {d.district.nunique()}
- Years: {d.year.min()}–{d.year.max()}
- Rainfall range: {f(rmin)}–{f(rmax)} mm
- Annual rice productivity range: {f(y_min)}–{f(y_max)} kg/ha
- Linear: {linear_text}
- Quadratic: {quad_text}
- Is β₂ negative? {'Yes' if b2<0 else 'No'}
- Statistically meaningful at 5%? {'Yes' if p2<.05 else 'No'}
- Turning point: {tp_text}
- Cluster-bootstrap conditional 95% interval: {f(result['bootstrap_low'])}–{f(result['bootstrap_high'])} mm ({result['bootstrap_valid_draws']} valid draws)
- Fixed effects: β₂={f(fe.params.iloc[2])}, {pvalue(fe.pvalues.iloc[2])} (district-clustered)
- Fixed-effects conclusion: {'supports negative curvature at 5%' if fe.params.iloc[2]<0 and fe.pvalues.iloc[2]<.05 else 'does not support a statistically significant inverted-U at 5%'}
- Robustness: {'negative curvature direction is consistent' if result['robust_direction'] else 'negative curvature direction is not consistent'}; {'all checks significant at 5%' if result['robust_significance'] else 'not all checks are significant at 5%'}
- Main conclusion: {conclusion}.
- Limitations: short stable-geography period, ambiguous season classification, district-level exposure measurement, omitted confounding and non-causal observational design.
"""
    (ROOT/"results/final_results.md").write_text(result_text)
    (AUDIT/"data_sources.md").write_text(f"""# Data provenance\n\nPipeline run (UTC): {datetime.now(timezone.utc).isoformat()}\n\n## Rice data\n- Dataset: ICAR-IIRR Rice Area, Production and Productivity Database\n- Organization: ICAR-Indian Institute of Rice Research\n- URL: {ICAR_URL}\n- File: {result['rice_meta'].get('input_file','user-supplied file')} (preserved in repository)\n- Years used: {d.year.min()}–{d.year.max()}\n- Variables: district, year, annual rice area (ha), production (tonnes), source yield (kg/ha).\n- Processing: header auto-detected; numeric QC; yield consistency calculated separately; unstable geography excluded.\n- Limitation: source description is not season-specific.\n\n## IMD rainfall\n- Dataset: 0.25-degree daily gridded rainfall NetCDF\n- Organization: India Meteorological Department\n- URL: {IMD_URL}; POST endpoint https://imdpune.gov.in/cmpg/Griddata/RF25.php (RF25=YEAR)\n- Years used: {d.year.min()}–{d.year.max()}\n- Variables/units: daily rainfall, source metadata examined; analysis totals in mm.\n- Processing: grid-cell centers within Census 2011 districts; daily available-cell means summed June–September; missingness not filled.\n- Limitations: grid support and district aggregation smooth local variability.\n\n## District boundaries\n- Dataset: Census 2011 district boundaries (DataMeet) via India Geodata repository\n- URL: {geo_info.get('boundary_url','https://github.com/yashveeeeeeer/india-geodata/tree/main/data/administrative/districts/census-2011')}\n- CRS: {geo_info.get('crs','EPSG:4326')}\n- Processing: exact matching with documented aliases only; Sri Potti Sriramulu Nellore is matched to SPSR Nellore, and Y.S.R. to the 2011 Kadapa polygon; no invented spatial crosswalk.\n- Limitation: 2011 boundaries approximate stable district units over the analysis period.\n\n## Geography decision\n{json.dumps(selection,indent=2)}\n""")


def write_readme() -> None:
    text="""# Rice Productivity and Monsoon Rainfall

This project studies how June–September rainfall is related to annual rice productivity in Andhra Pradesh. It also includes a small app for exploring yield estimates under rainfall scenarios. The study is observational, so it cannot show that rainfall caused a change in yield.

## Data

The analysis combines district rice data from [ICAR-IIRR](https://icar-iirr.org/index.php/en/component/content/article/39-iirr-databases/214-rice-area-production-and-productivity-database), daily rainfall from the [India Meteorological Department](https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html), and 2011 district boundaries from [India Geodata](https://github.com/yashveeeeeeer/india-geodata/tree/main/data/administrative/districts/census-2011). The rice database description does not say which crop season its figures cover, so this project calls them annual rice productivity.

District boundaries changed over time. To keep the comparisons consistent, the study uses the longest period with a complete set of unchanged Andhra Pradesh districts. This covers 2013–2021.

## Setup and run

Python 3.11 or later is recommended. From the project folder, install the listed packages and run the analysis:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

The first run needs internet access to download rainfall and boundary data. The pipeline finds the supplied rice CSV in the project, processes the data, runs the analysis, and creates the tables, charts, and reports.

To start the prediction app, train its models and then launch Streamlit:

```bash
python -m src.train_models
streamlit run app.py
```

The app estimates annual rice yield from rainfall and district. Cultivated area is optional and is used to estimate production. The model does not forecast rainfall or provide a guaranteed forecast of crop results. Its evaluation uses only 117 records from 13 districts, and it performs poorly when asked to predict for districts it has not seen.

## Main files and outputs

- `run_pipeline.py`: runs the research analysis
- `app.py`: starts the prediction app
- `src/`: data processing, analysis, model, and chart code
- `tests/`: checks for the data pipeline and app components
- `requirements.txt`: Python packages used by the project
- `data/processed/`: cleaned and matched data
- `results/tables/`: model estimates and data checks
- `results/data_audit/`: data sources and processing decisions
- `results/final_results.md`: short summary of the research findings
- `report/research_report.md`: full research report
- `figures/`: charts from the analysis

The research analysis uses HC3 standard errors for the pooled models and district-clustered standard errors for the district and year fixed-effects model. Missing yield and rainfall values are not filled in. The bootstrap uses a fixed random seed (20261008) so its results can be reproduced. Downloaded source data are kept under `data/raw/` and are not changed by the analysis.

## Limits of the study

The rice data do not identify the crop season. District-level rainfall averages can hide local differences and rainfall timing. The study does not account for factors such as irrigation, soil, temperature, or crop management. Its findings describe associations in this dataset and should not be treated as a universal rainfall target or a causal result.
"""
    (ROOT/"README.md").write_text(text)


def main() -> None:
    for p in (PROCESSED, TABLES, AUDIT, FIG/"eda", FIG/"models", FIG/"diagnostics", RAW/"imd_rainfall"):
        p.mkdir(parents=True, exist_ok=True)
    source=discover_rice_file(); rice, rice_meta=read_rice(source)
    rice.to_csv(PROCESSED/"rice_cleaned.csv",index=False)
    study, selection=select_stable_ap(rice)
    years=sorted(map(int,study.year.unique()))
    files=download_years(years,RAW/"imd_rainfall")
    boundary=obtain_boundaries(RAW/"boundaries")
    rainfall,geo_meta=district_rainfall(files,sorted(study.analysis_district.unique()),boundary)
    study=study.drop(columns=["analysis_district"])
    matched=merge_district_year(study,rainfall)
    merge_meta={"rice_rows_in_stable_panel":len(study),"rainfall_rows":len(rainfall),"matched_rows":len(matched),"unmatched_rice_rows":int(len(study)-len(matched)),"unmatched_rainfall_rows":int(len(rainfall)-len(matched)),"merge_validation":"one-to-one district-year"}
    (TABLES/"05_merge_summary.csv").write_text(pd.DataFrame([merge_meta]).to_csv(index=False))
    result=run_analysis(matched,rice_meta,geo_meta); result["rice_meta"]=rice_meta
    write_report(result,geo_meta,selection); write_readme()
    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=False)
    if test_run.returncode:
        raise RuntimeError(f"Pipeline tests failed with exit code {test_run.returncode}")
    # Explicit quality-control artifact and hard failure for incomplete mandatory outputs.
    expected=[ROOT/"README.md",ROOT/"report/research_report.md",ROOT/"results/final_results.md",ROOT/"results/data_audit/data_sources.md",ROOT/"data/processed/rice_cleaned.csv",ROOT/"data/processed/analysis_dataset.csv"]
    expected += [TABLES/f"{n:02d}_{name}.csv" for n,name in [(1,"data_summary"),(2,"missingness"),(3,"rice_cleaning_summary"),(4,"rainfall_summary"),(5,"merge_summary"),(6,"linear_model"),(7,"quadratic_model"),(8,"model_comparison"),(9,"turning_point"),(10,"bootstrap_turning_point"),(11,"fixed_effects"),(12,"robustness"),(13,"influence_diagnostics")]]
    expected += [FIG/sub/f"{n:02d}_{name}.png" for sub,entries in [("eda",[(1,"yield_distribution"),(2,"rainfall_distribution"),(3,"rainfall_vs_yield"),(4,"yearly_rainfall"),(5,"yearly_yield"),(6,"rainfall_bins")]),("models",[(7,"linear_fit"),(8,"quadratic_fit"),(9,"bootstrap_turning_point"),(10,"fixed_effects_relationship")]),("diagnostics",[(11,"residual_fitted"),(12,"qq_plot"),(13,"influence_diagnostics")])] for n,name in entries]
    missing=[str(p) for p in expected if not p.exists() or p.stat().st_size==0]
    if missing: raise RuntimeError(f"Pipeline output quality check failed: {missing}")
    (AUDIT/"pipeline_quality_control.json").write_text(json.dumps({"source":str(source.relative_to(ROOT)),"rice_meta":rice_meta,"geography":selection,"rainfall":geo_meta,"merge":merge_meta,"outputs_verified":True,"expected_outputs":len(expected),"pytest_return_code":test_run.returncode},indent=2,default=str))
    print(f"Pipeline complete: {len(result['data'])} matched district-years; report: report/research_report.md")


if __name__ == "__main__":
    main()
