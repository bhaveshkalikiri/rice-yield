"""Plotly charts for the RiceYield AI dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#243B64"
CREAM = "#FAF3E0"
GREEN = BLUE
PALETTE = [BLUE]


def rainfall_yield_scatter(data: pd.DataFrame, selected_district: str | None = None, scenario: dict | None = None):
    if selected_district:
        data = data.loc[data["district"].eq(selected_district)].copy()
    fig = px.scatter(
        data, x="monsoon_rainfall_mm", y="yield_kg_ha", color="district",
        hover_data=["year", "district"], color_discrete_sequence=PALETTE,
        labels={"monsoon_rainfall_mm": "June–September rainfall (mm)", "yield_kg_ha": "Annual rice productivity (kg/ha)", "district": "District"},
        opacity=.7,
    )
    if selected_district:
        fig.update_traces(selector=dict(name=selected_district), marker={"size": 11, "line": {"width": 1, "color": BLUE}})
    if scenario:
        fig.add_trace(go.Scatter(
            x=[scenario["rainfall_mm"]], y=[scenario["predicted_yield_kg_ha"]], mode="markers+text",
            name="Model scenario prediction", text=["Scenario prediction"], textposition="top center",
            marker={"size": 17, "color": CREAM, "symbol": "star", "line": {"width": 2, "color": BLUE}},
        ))
    fig.update_layout(template="none", height=390, legend_title_text="District", margin={"l": 10, "r": 10, "t": 25, "b": 10}, plot_bgcolor=CREAM, paper_bgcolor=CREAM, font={"color":BLUE, "family":"Inter, sans-serif"}, xaxis={"gridcolor":"rgba(36,59,100,.18)"}, yaxis={"gridcolor":"rgba(36,59,100,.18)"})
    return fig


def yearly_yield_chart(data: pd.DataFrame):
    yearly = data.groupby("year", as_index=False).agg(mean_yield=("yield_kg_ha", "mean"), sem=("yield_kg_ha", "sem"))
    fig = px.line(yearly, x="year", y="mean_yield", markers=True, labels={"year": "Year", "mean_yield": "Mean annual rice productivity (kg/ha)"}, color_discrete_sequence=[GREEN])
    fig.add_scatter(x=yearly.year, y=yearly.mean_yield, mode="markers", marker={"color": GREEN, "size": 8}, error_y={"type": "data", "array": 1.96 * yearly["sem"], "visible": True}, showlegend=False)
    fig.update_layout(template="none", height=350, margin={"l": 10, "r": 10, "t": 20, "b": 10}, plot_bgcolor=CREAM, paper_bgcolor=CREAM, font={"color":BLUE, "family":"Inter, sans-serif"}, xaxis={"gridcolor":"rgba(36,59,100,.18)"}, yaxis={"gridcolor":"rgba(36,59,100,.18)"})
    return fig


def rainfall_distribution(data: pd.DataFrame):
    fig = px.histogram(data, x="monsoon_rainfall_mm", nbins=18, color_discrete_sequence=[BLUE], labels={"monsoon_rainfall_mm": "June–September rainfall (mm)", "count": "District-years"})
    fig.update_layout(template="none", height=330, margin={"l": 10, "r": 10, "t": 20, "b": 10}, plot_bgcolor=CREAM, paper_bgcolor=CREAM, font={"color":BLUE, "family":"Inter, sans-serif"}, xaxis={"gridcolor":"rgba(36,59,100,.18)"}, yaxis={"gridcolor":"rgba(36,59,100,.18)"})
    return fig


def district_comparison(data: pd.DataFrame):
    summary = data.groupby("district", as_index=False).agg(mean_yield=("yield_kg_ha", "mean")).sort_values("mean_yield")
    fig = px.bar(summary, x="mean_yield", y="district", orientation="h", color="mean_yield", color_continuous_scale=[CREAM, BLUE], labels={"mean_yield": "Mean annual rice productivity (kg/ha)", "district": "District"})
    fig.update_layout(template="none", height=470, coloraxis_showscale=False, margin={"l": 10, "r": 10, "t": 20, "b": 10}, plot_bgcolor=CREAM, paper_bgcolor=CREAM, font={"color":BLUE, "family":"Inter, sans-serif"}, xaxis={"gridcolor":"rgba(36,59,100,.18)"})
    return fig


def model_error_chart(metrics: pd.DataFrame):
    fig = px.bar(metrics.sort_values("mae_kg_ha"), x="mae_kg_ha", y="model", orientation="h", color="model", color_discrete_sequence=PALETTE, labels={"mae_kg_ha": "Chronological holdout MAE (kg/ha)", "model": "Model"})
    fig.update_layout(template="none", showlegend=False, height=330, margin={"l": 10, "r": 10, "t": 20, "b": 10}, plot_bgcolor=CREAM, paper_bgcolor=CREAM, font={"color":BLUE, "family":"Inter, sans-serif"}, xaxis={"gridcolor":"rgba(36,59,100,.18)"})
    return fig


def importance_chart(names, values, title="Feature importance"):
    frame = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance")
    fig = px.bar(frame, x="importance", y="feature", orientation="h", color_discrete_sequence=[GREEN], labels={"importance": title, "feature": "Feature"})
    fig.update_layout(template="none", height=max(300, 30 * len(frame)), margin={"l": 10, "r": 10, "t": 20, "b": 10}, plot_bgcolor=CREAM, paper_bgcolor=CREAM, font={"color":BLUE, "family":"Inter, sans-serif"}, xaxis={"gridcolor":"rgba(36,59,100,.18)"})
    return fig
