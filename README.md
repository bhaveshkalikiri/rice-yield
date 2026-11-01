# Rice Productivity and Monsoon Rainfall

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
