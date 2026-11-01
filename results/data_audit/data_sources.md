# Data provenance

Pipeline run (UTC): 2026-10-08T10:26:05.862819+00:00

## Rice data
- Dataset: ICAR-IIRR Rice Area, Production and Productivity Database
- Organization: ICAR-Indian Institute of Rice Research
- URL: https://icar-iirr.org/index.php/en/component/content/article/39-iirr-databases/214-rice-area-production-and-productivity-database
- File: Rice_Data_2026-10-08 (1).csv (preserved in repository)
- Years used: 2013–2021
- Variables: district, year, annual rice area (ha), production (tonnes), source yield (kg/ha).
- Processing: header auto-detected; numeric QC; yield consistency calculated separately; unstable geography excluded.
- Limitation: source description is not season-specific.

## IMD rainfall
- Dataset: 0.25-degree daily gridded rainfall NetCDF
- Organization: India Meteorological Department
- URL: https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html; POST endpoint https://imdpune.gov.in/cmpg/Griddata/RF25.php (RF25=YEAR)
- Years used: 2013–2021
- Variables/units: daily rainfall, source metadata examined; analysis totals in mm.
- Processing: grid-cell centers within Census 2011 districts; daily available-cell means summed June–September; missingness not filled.
- Limitations: grid support and district aggregation smooth local variability.

## District boundaries
- Dataset: Census 2011 district boundaries (DataMeet) via India Geodata repository
- URL: https://github.com/yashveeeeeeer/india-geodata/tree/main/data/administrative/districts/census-2011
- CRS: EPSG:4326
- Processing: exact matching with documented aliases only; Sri Potti Sriramulu Nellore is matched to SPSR Nellore, and Y.S.R. to the 2011 Kadapa polygon; no invented spatial crosswalk.
- Limitation: 2011 boundaries approximate stable district units over the analysis period.

## Geography decision
{
  "selection_rule": "longest contiguous years with invariant Andhra Pradesh rice district roster and complete district-year reporting",
  "analysis_year_start": 2013,
  "analysis_year_end": 2021,
  "district_count": 13,
  "included_districts": [
    "ANANTAPUR",
    "CHITTOOR",
    "EAST GODAVARI",
    "GUNTUR",
    "KRISHNA",
    "KURNOOL",
    "PRAKASAM",
    "SPSR NELLORE",
    "SRIKAKULAM",
    "VISAKHAPATANAM",
    "VIZIANAGARAM",
    "WEST GODAVARI",
    "Y.S.R."
  ],
  "ap_year_district_counts": {
    "1997": 23,
    "1998": 23,
    "1999": 23,
    "2000": 23,
    "2001": 22,
    "2002": 22,
    "2003": 22,
    "2004": 22,
    "2005": 22,
    "2006": 22,
    "2007": 22,
    "2008": 22,
    "2009": 22,
    "2010": 22,
    "2011": 21,
    "2012": 21,
    "2013": 13,
    "2014": 13,
    "2015": 13,
    "2016": 13,
    "2017": 13,
    "2018": 13,
    "2019": 13,
    "2020": 13,
    "2021": 13,
    "2022": 26,
    "2023": 26
  },
  "excluded_ap_rows": 406,
  "excluded_years_boundary_change": [
    1997,
    1998,
    1999,
    2000,
    2001,
    2002,
    2003,
    2004,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2022,
    2023
  ]
}
