"""
frontend_app.py

Interactive dashboard for the Food Delivery Analytics project, built
with Streamlit. Unlike the static outputs/dashboard.html (which is a
frozen snapshot), this one lets the user actively filter the data by
city, weather and traffic level, and every chart/metric/insight below
recalculates live from whatever subset of orders is currently selected.

It deliberately reuses the same analysis functions from src/analysis.py
that the main pipeline uses - the filtering logic lives here, but the
actual number-crunching (traffic_impact, distance_impact, etc.) is not
duplicated.

Run with:
    streamlit run frontend_app.py

Author: Data Engineering Team
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from src import analysis

DATA_PATH = "data/food_delivery_dataset.csv"

st.set_page_config(page_title="Food Delivery Analytics", layout="wide")


@st.cache_data
def get_cleaned_data() -> pd.DataFrame:
    """
    Load and clean the dataset once per session. Cached so switching
    filters doesn't re-read and re-clean 39,000 rows from disk on
    every single interaction.
    """
    raw = analysis.load_dataset(DATA_PATH)
    return analysis.clean_dataset(raw)


def plot_traffic_bar(traffic_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        traffic_df["Road_traffic_density"].astype(str),
        traffic_df["avg_delivery_time"],
        color="#3B6FA0",
    )
    for bar, value in zip(bars, traffic_df["avg_delivery_time"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Average Delivery Time by Traffic Density")
    ax.set_xlabel("Traffic Density")
    ax.set_ylabel("Avg Delivery Time (min)")
    fig.tight_layout()
    return fig


def plot_distance_scatter(df: pd.DataFrame):
    clean = df.dropna(subset=["distance_km", "Time_taken_min"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(clean["distance_km"], clean["Time_taken_min"], alpha=0.15, s=10, color="#C0562F")

    if len(clean) > 1:
        binned = clean.copy()
        binned["_bin"] = pd.cut(binned["distance_km"], bins=15)
        avgs = binned.groupby("_bin", observed=True).agg(
            mid=("distance_km", "mean"), avg_time=("Time_taken_min", "mean")
        )
        ax.plot(avgs["mid"], avgs["avg_time"], color="#1B1B1B", linewidth=2, label="Binned average")
        ax.legend()

    ax.set_title("Distance vs Delivery Time")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (min)")
    fig.tight_layout()
    return fig


def plot_weather_heatmap(combo_df: pd.DataFrame):
    pivot = combo_df.pivot(
        index="Weather_conditions", columns="Road_traffic_density", values="avg_delivery_time"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9)

    ax.set_title("Weather x Traffic - Avg Delivery Time")
    fig.colorbar(im, ax=ax, label="min")
    fig.tight_layout()
    return fig


def main():
    st.title("Food Delivery Analytics Dashboard")
    st.caption("Use the filters on the left to explore how traffic, distance and weather affect delivery time.")

    df = get_cleaned_data()

    st.sidebar.header("Filters")
    cities = sorted(df["City"].dropna().unique().tolist())
    weathers = sorted(df["Weather_conditions"].dropna().unique().tolist())
    traffic_levels = [t for t in analysis.TRAFFIC_ORDER if t in df["Road_traffic_density"].unique()]

    selected_cities = st.sidebar.multiselect("City", cities, default=cities)
    selected_weather = st.sidebar.multiselect("Weather", weathers, default=weathers)
    selected_traffic = st.sidebar.multiselect("Traffic Density", traffic_levels, default=traffic_levels)

    filtered = df[
        df["City"].isin(selected_cities)
        & df["Weather_conditions"].isin(selected_weather)
        & df["Road_traffic_density"].astype(str).isin(selected_traffic)
    ]

    if filtered.empty:
        st.warning("No orders match the selected filters. Try widening your selection.")
        return

    summary = analysis.basic_summary(filtered)

    col1, col2, col3 = st.columns(3)
    col1.metric("Orders", f"{summary['total_orders']:,}")
    col2.metric("Avg Delivery Time", f"{summary['avg_delivery_time_min']} min")
    col3.metric("Avg Distance", f"{summary['avg_distance_km']} km")

    traffic_df = analysis.traffic_impact(filtered)
    distance_df, distance_corr = analysis.distance_impact(filtered)
    combo_df = analysis.weather_traffic_combo(filtered)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(plot_traffic_bar(traffic_df))
    with chart_col2:
        st.pyplot(plot_distance_scatter(filtered))

    st.pyplot(plot_weather_heatmap(combo_df))

    st.subheader("Business Insights (for the current filter selection)")
    insights = analysis.generate_insights(traffic_df, distance_corr, combo_df)
    for point in insights:
        st.markdown(f"- {point}")


if __name__ == "__main__":
    main()