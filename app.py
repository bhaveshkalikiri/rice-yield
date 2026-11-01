from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ARTIFACT_PATH, DATA_PATH, METADATA_PATH, METRICS_PATH
from src.data_loader import load_dataset
from src.prediction import load_model_artifact, normalize_prediction_result, predict_scenario
from src.visualizations import district_comparison, model_error_chart, rainfall_distribution, rainfall_yield_scatter, yearly_yield_chart

st.set_page_config(page_title="RiceYield AI", page_icon="🔷", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#FAF3E0; color:#243B64; }
[data-testid="stMainBlockContainer"] { max-width:1180px; padding:2rem 2rem 1rem; }
html, body, [class*="css"] { font-family:Inter, "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
h1,h2,h3,h4 { font-family:"DM Sans", Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#243B64; letter-spacing:-.025em; }
h1 { font-size:2.5rem; line-height:1.12; } h2 { font-size:1.65rem; } h3 { font-size:1.2rem; }
p, li, label, [data-testid="stCaptionContainer"] { color:#243B64; line-height:1.55; }
[data-testid="stCaptionContainer"] { font-size:.9rem; }
[data-testid="stSidebar"] { background:#FAF3E0; border-right:1px solid #243B64; }
[data-testid="stHeader"] { background:#FAF3E0; }
.hero { background:#FAF3E0; border:1px solid #243B64; border-radius:22px; padding:26px 32px; margin:0 0 20px; }
.eyebrow { color:#243B64; font-size:.73rem; font-weight:700; letter-spacing:.13em; margin:0 0 8px; }
.hero h1 { margin:0 0 8px; color:#243B64; }
.hero p { margin:0; color:#243B64; font-size:1.08rem; }
.subtle-note { color:#243B64; font-size:.88rem; margin:.2rem 0 1.2rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#243B64 !important; border-radius:16px !important; background:#FAF3E0; }
.st-key-scenario_panel, .st-key-estimate_panel { height:610px; min-height:610px; overflow-y:auto; }
[data-testid="stMetric"] { background:#FAF3E0; border:1px solid #243B64; border-radius:14px; padding:16px 18px; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:#243B64 !important; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button { background:#243B64; color:#FAF3E0; border:1px solid #243B64; border-radius:11px; min-height:3.15rem; font-size:1rem; font-weight:700; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background:#FAF3E0; color:#243B64; border:2px solid #243B64; }
[data-testid="stDownloadButton"] button { background:#FAF3E0; color:#243B64; border:1px solid #243B64; }
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { background:#FAF3E0; color:#243B64; border-color:#243B64; border-radius:9px; }
[data-baseweb="select"] *, [data-testid="stNumberInput"] button { color:#243B64; }
[data-baseweb="popover"], [role="listbox"], [role="option"] { background:#FAF3E0; color:#243B64; }
[data-testid="stWidgetLabel"] p { color:#243B64; font-weight:700; }
[data-testid="stSlider"] [role="slider"] { background:#243B64; }
[data-testid="stRadio"] label { color:#243B64; }
[data-testid="stDataFrame"] { border:1px solid #243B64; border-radius:12px; overflow:hidden; background:#FAF3E0; }
[data-testid="stExpander"] { border:1px solid #243B64; border-radius:12px; background:#FAF3E0; }
[data-testid="stAlert"] { border:1px solid #243B64; border-radius:12px; background:#FAF3E0; color:#243B64; }
[data-testid="stAlert"] * { color:#243B64 !important; }
.category { display:inline-block; border-radius:999px; padding:7px 13px; font-size:.82rem; font-weight:800; letter-spacing:.07em; color:#243B64; border:1px solid #243B64; }
.category-good, .category-moderate, .category-poor { background:#FAF3E0; }
.footer { border-top:1px solid #243B64; margin-top:2.3rem; padding:1rem 0 .5rem; color:#243B64; font-size:.85rem; }
.footer strong { color:#243B64; }
@media (max-width:700px) { [data-testid="stMainBlockContainer"] { padding:1rem 1rem .5rem; } .hero { padding:20px; border-radius:17px; } h1 { font-size:2rem; } .st-key-scenario_panel, .st-key-estimate_panel { min-height:auto; } }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_artifact(path: str, modified_ns: int):
    return load_model_artifact(path)


@st.cache_data(show_spinner=False)
def get_metrics(path: str, modified_ns: int):
    return pd.read_csv(path)


try:
    data = get_data()
    artifact = get_artifact(str(ARTIFACT_PATH), ARTIFACT_PATH.stat().st_mtime_ns)
    metrics = get_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime_ns)
    metadata = json.loads(METADATA_PATH.read_text())
except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
    st.error(f"RiceYield AI could not load its dataset or trained model: {exc}")
    st.caption("From the project folder, retrain the model with `.venv/bin/python -m src.train_models`.")
    st.stop()

st.markdown('<header class="hero"><div class="eyebrow">AGRICULTURE ANALYTICS · ANDHRA PRADESH</div><h1>RiceYield AI</h1><p>Understand rice productivity. Explore rainfall scenarios.</p></header>', unsafe_allow_html=True)
st.markdown('<div class="subtle-note">Scenario estimates based on historical observations · Not official agricultural forecasts</div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### RiceYield AI")
    page = st.selectbox("View", ["Prediction", "Explore historical data", "Model performance", "About the project"], label_visibility="collapsed")

if page == "Prediction":
    st.subheader("Predict rice productivity")
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True, key="scenario_panel"):
            st.markdown("#### Set your scenario")
            with st.form("scenario_form"):
                district = st.selectbox("District", artifact["districts"], help="Choose a district represented in historical training data.")
                rainfall = st.number_input("June–September monsoon rainfall (mm)", min_value=0.0, max_value=2500.0, value=float(np.median(data.monsoon_rainfall_mm)), step=10.0, help="Enter a rainfall scenario. The app does not forecast rainfall.")
                area = st.number_input("Cultivated area (hectares)", min_value=0.01, value=None, step=10.0, help="Optional. Leave blank if you only want yield per hectare.")
                submitted = st.form_submit_button("Predict Rice Yield", type="primary", width="stretch")
            st.caption("Predictions use district identity and rainfall from the historical study period. Area is optional.")
    with right:
        if "scenario_prediction" in st.session_state:
            try:
                result = normalize_prediction_result(st.session_state["scenario_prediction"], artifact["category_thresholds_kg_ha"])
                st.session_state["scenario_prediction"] = result
            except ValueError as exc:
                st.session_state.pop("scenario_prediction", None)
                st.warning(str(exc))
                result = None
        else:
            result = None
        if submitted:
            st.session_state.pop("scenario_prediction", None)
            try:
                result = predict_scenario(artifact, district, rainfall, area)
                st.session_state["scenario_prediction"] = result
            except ValueError as exc:
                result = None
                st.error(str(exc))
        if result:
            with st.container(border=True, key="estimate_panel"):
                st.markdown("#### Scenario estimate")
                st.metric("Predicted rice yield", f"{result['predicted_yield_kg_ha']:,.0f} kg/ha")
                m1, m2 = st.columns(2)
                m1.metric("Tonnes per hectare", f"{result['predicted_yield_tonnes_ha']:.2f} t/ha")
                category_class = {"GOOD":"category-good", "MODERATE":"category-moderate", "POOR":"category-poor"}[result["productivity_category"]]
                m2.markdown(f"<div style='color:#243B64;font-size:.9rem;font-weight:600;margin:0 0 8px'>Productivity category · relative band</div><span class='category {category_class}'>{result['productivity_category']}</span>", unsafe_allow_html=True)
                st.caption(f"Bands use historical yield quartiles: POOR < {artifact['category_thresholds_kg_ha']['q25_kg_ha']:,.0f}; MODERATE below {artifact['category_thresholds_kg_ha']['q75_kg_ha']:,.0f}; GOOD at or above {artifact['category_thresholds_kg_ha']['q75_kg_ha']:,.0f} kg/ha. These are not crop-success probabilities.")
                ref1, ref2 = st.columns(2)
                ref1.metric("District historical mean", f"{result['historical_reference_yield_kg_ha']:,.0f} kg/ha" if result["historical_reference_yield_kg_ha"] is not None else "Unavailable")
                ref2.metric("Difference from mean", f"{result['difference_from_reference_kg_ha']:+,.0f} kg/ha" if result["difference_from_reference_kg_ha"] is not None else "Unavailable", f"{result['percentage_difference_from_reference']:+.1f}%" if result["percentage_difference_from_reference"] is not None else None)
                if result["estimated_production_tonnes"] is not None:
                    st.metric(f"Estimated production · {result['area_ha']:,.1f} ha", f"{result['estimated_production_tonnes']:,.1f} tonnes")
                if result["extrapolation_warning"]:
                    st.warning(result["extrapolation_warning"])
                if result["prediction_clipped"]:
                    st.warning(next((w for w in result["warnings"] if "clipped" in w), "Negative raw yield was clipped to zero; treat this scenario as a model failure."))
        else:
            with st.container(border=True, key="estimate_panel"):
                st.markdown("#### Your estimate will appear here")
                st.write("Choose a district and enter an expected rainfall scenario, then select **Predict Rice Yield**.")
                st.caption("The result reports annual rice yield in kg/ha and t/ha, with an optional production estimate.")

    if result:
        st.markdown("### Historical context")
        hist_col, ref_col = st.columns([1.2, .8], gap="large")
        hist_col.plotly_chart(rainfall_yield_scatter(data, result["district"], result), width="stretch")
        yearly = data.loc[data.district.eq(result["district"])].sort_values("year")
        ref_col.plotly_chart(yearly_yield_chart(yearly), width="stretch")
        compare = pd.DataFrame({"Measure": ["Historical district mean", "Scenario model estimate"], "Yield (kg/ha)": [result["historical_reference_yield_kg_ha"], result["predicted_yield_kg_ha"]]})
        fig = px.bar(compare, x="Measure", y="Yield (kg/ha)", color="Measure", color_discrete_map={"Historical district mean":"#FAF3E0", "Scenario model estimate":"#243B64"})
        fig.update_traces(marker_line_color="#243B64", marker_line_width=1.5)
        fig.update_layout(template="none", showlegend=False, height=300, plot_bgcolor="#FAF3E0", paper_bgcolor="#FAF3E0", font={"color":"#243B64", "family":"Inter, sans-serif"}, xaxis={"gridcolor":"rgba(36,59,100,.18)"}, yaxis={"gridcolor":"rgba(36,59,100,.18)"})
        st.plotly_chart(fig, width="stretch")
    with st.expander("How to interpret this scenario"):
        st.write("This is a scenario estimate conditional on the rainfall you entered, not a rainfall or yield forecast. Rainfall alone cannot explain yield: irrigation, soils, fertilizer, temperature, variety and management data are not available in this model. No validated prediction interval is provided.")

elif page == "Explore historical data":
    st.subheader("Explore historical data")
    st.caption("Observed annual rice yield and June–September rainfall in the merged district-year panel.")
    c1, c2 = st.columns(2)
    districts = sorted(data.district.unique())
    selected = c1.multiselect("Districts", districts, default=districts)
    years = c2.slider("Study years", int(data.year.min()), int(data.year.max()), (int(data.year.min()), int(data.year.max())))
    view = data.loc[data.district.isin(selected) & data.year.between(*years)].copy()
    if view.empty:
        st.info("No observations match these filters.")
    else:
        st.caption(f"{len(view)} district-year observations")
        a, b = st.columns(2, gap="large")
        a.plotly_chart(rainfall_yield_scatter(view), width="stretch")
        b.plotly_chart(district_comparison(view), width="stretch")
        c, d = st.columns(2, gap="large")
        c.plotly_chart(yearly_yield_chart(view), width="stretch")
        d.plotly_chart(rainfall_distribution(view), width="stretch")
        st.dataframe(view[["district", "year", "monsoon_rainfall_mm", "yield_kg_ha", "area_ha", "production_tonnes"]].sort_values(["district", "year"]), width="stretch", hide_index=True)

elif page == "Model performance":
    st.subheader("Model performance")
    st.markdown(f"**Selected model: {artifact['selected_model_name']}** · selection rule: {metadata['selection_rule']}")
    st.caption("Metrics below come from held-out evaluation, not the production model refit on all observations. Grouped district validation is reported separately and assesses transfer to unseen districts.")
    columns = ["model", "mae_kg_ha", "rmse_kg_ha", "r2", "validation_method", "training_n", "validation_n", "district_grouped_mae_kg_ha", "district_grouped_rmse_kg_ha", "district_grouped_r2", "district_grouped_validation_n_oof"]
    display = metrics[columns].rename(columns={"model":"Model", "mae_kg_ha":"MAE (kg/ha)", "rmse_kg_ha":"RMSE (kg/ha)", "r2":"R²", "validation_method":"Validation strategy", "training_n":"Train n", "validation_n":"Validation n", "district_grouped_mae_kg_ha":"District CV MAE", "district_grouped_rmse_kg_ha":"District CV RMSE", "district_grouped_r2":"District CV R²", "district_grouped_validation_n_oof":"District CV n"})
    st.dataframe(display.style.format({"MAE (kg/ha)":"{:.1f}", "RMSE (kg/ha)":"{:.1f}", "R²":"{:.3f}", "District CV MAE":"{:.1f}", "District CV RMSE":"{:.1f}", "District CV R²":"{:.3f}"}), width="stretch", hide_index=True)
    st.plotly_chart(model_error_chart(metrics), width="stretch")
    st.warning("The sample is small. The baseline outperforms some candidates on RMSE/R², and grouped-district R² is negative; transfer to unseen districts is weak. These metrics are not accuracy percentages.")

elif page == "About the project":
    st.subheader("About RiceYield AI")
    st.write(f"The application uses {artifact['observation_count']} annual district observations from {artifact['year_min']}–{artifact['year_max']} across {artifact['district_count']} districts. The production model was fitted on all {artifact['final_training_n']} eligible rows.")
    st.write("The historical panel is small and rainfall alone does not explain yield variation. Irrigation, soil, fertilizer, temperature, crop variety and management are not model inputs. Predictions are scenario estimates based on historical observations, not guaranteed outcomes or official agricultural forecasts.")
    st.write("GOOD, MODERATE and POOR are relative historical productivity bands defined using the full eligible yield distribution: below Q25, Q25 to below Q75, and Q75 or higher. They are not official classifications or probabilities of crop success.")
    st.info(f"Saved thresholds: Q25 = {artifact['category_thresholds_kg_ha']['q25_kg_ha']:,.1f} kg/ha; Q75 = {artifact['category_thresholds_kg_ha']['q75_kg_ha']:,.1f} kg/ha.")

st.markdown('<footer class="footer"><strong>RiceYield AI</strong> · A data-driven exploration of rice productivity in Andhra Pradesh.<br>Predictions are scenario estimates based on historical observations. They are not official agricultural forecasts.</footer>', unsafe_allow_html=True)
