import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import (
    read_rice, select_stable_ap, bootstrap_turning_point,
    aggregate_monsoon, merge_district_year, quadratic_turning_point,
    validate_rainfall_year,
)


def test_metadata_line_removed_and_csv_parsed(tmp_path):
    p = tmp_path / "rice.csv"
    p.write_text('<!-- FILE: config.php -->\nState,District,Year,"Area (Ha)","Production (Tonnes)","Yield (Kg/ha)"\nANDHRA PRADESH,TEST,2019,10,20,2000\n')
    data, meta = read_rice(p)
    assert len(data) == 1
    assert data.iloc[0].yield_calculated_kg_ha == 2000
    assert meta["source_header_line_1based"] == 2


def test_duplicate_count_and_numeric_conversion(tmp_path):
    p = tmp_path / "rice.csv"
    p.write_text('State,District,Year,Area,Production,Yield\nANDHRA PRADESH,A,2019,"1,000",2000,2000\nANDHRA PRADESH,A,2019,1000,2000,2000\n')
    data, meta = read_rice(p)
    assert data.area_ha.tolist() == [1000.0, 1000.0]
    assert meta["duplicate_district_year_records"] == 2


def test_stable_panel_selects_longest_invariant_geography():
    rows = []
    for year in range(2013, 2022):
        for district in ["A", "B"]:
            rows.append({"state": "ANDHRA PRADESH", "district": district, "year": year, "area_ha": 10, "production_tonnes": 20, "yield_kg_ha": 2000})
    rows.append({"state": "ANDHRA PRADESH", "district": "NEW", "year": 2022, "area_ha": 10, "production_tonnes": 20, "yield_kg_ha": 2000})
    out, info = select_stable_ap(pd.DataFrame(rows))
    assert out.year.min() == 2013 and out.year.max() == 2021
    assert info["district_count"] == 2


def test_bootstrap_reproducible():
    import src.pipeline as p
    rows = []
    for district in ["A", "B", "C"]:
        for year, rain in enumerate([500, 700, 900, 1100], 2015):
            rows.append({"district": district, "year": year, "monsoon_rainfall_mm": rain, "yield_kg_ha": 2000 + 500 * (rain / 100) - 25 * (rain / 100) ** 2 + (ord(district) - 65) * 3, "rainfall_100mm": rain / 100})
    data = pd.DataFrame(rows)
    first = p.bootstrap_turning_point(data, draws=25)[3]
    second = p.bootstrap_turning_point(data, draws=25)[3]
    pd.testing.assert_frame_equal(first, second)


def test_june_september_filter_and_leap_day():
    dates = pd.date_range("2020-01-01", "2020-12-31")
    mask = (dates.month >= 6) & (dates.month <= 9)
    assert mask.sum() == 122
    assert pd.Timestamp("2020-02-29") not in dates[mask]


def test_rainfall_year_filtering_rejects_mixed_or_wrong_year():
    dates = pd.date_range("2019-01-01", "2019-12-31")
    validate_rainfall_year(dates, 2019)
    try:
        validate_rainfall_year(dates, 2020)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-year rainfall data should fail validation")


def test_june_september_aggregation_and_leap_year():
    dates = pd.date_range("2020-01-01", "2020-12-31")
    values = np.full((len(dates), 2), 1.0)
    assert aggregate_monsoon(values, dates, np.array([0, 1]), 2020) == 122


def test_district_year_merge_integrity():
    rice = pd.DataFrame({"district": ["A", "B"], "year": [2020, 2020], "yield_kg_ha": [1, 2]})
    rain = pd.DataFrame({"district": ["A", "B"], "year": [2020, 2020], "monsoon_rainfall_mm": [10, 20]})
    assert len(merge_district_year(rice, rain)) == 2
    with_duplicate = pd.concat([rain, rain.iloc[[0]]], ignore_index=True)
    try:
        merge_district_year(rice, with_duplicate)
    except pd.errors.MergeError:
        pass
    else:
        raise AssertionError("duplicate district-year rainfall must fail the merge")


def test_interior_quadratic_turning_point_only_for_negative_curvature():
    point, inside = quadratic_turning_point(20, -1, 0, 3000)
    assert point == 1000 and inside
    point, inside = quadratic_turning_point(20, 1, 0, 3000)
    assert np.isnan(point) and not inside
