

import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow running these tests directly with `pytest` from the project
# root without needing to install the package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import analysis


@pytest.fixture
def sample_df():
    """A tiny, hand-crafted dataset covering the edge cases we care about."""
    return pd.DataFrame(
        {
            "Time_taken (min)": [15, 45, 35, 20, 25, 50],
            "distance_km": [2.0, 9.0, 7.0, 3.0, 5.0, 0.0],
            "Delivery_person_Age": [25, 40, 150, 22, None, 30],
            "Delivery_person_Ratings": [4.5, 3.8, 9.9, 4.0, None, 4.2],
            "Weather_conditions": [" Sunny", "Stormy", "Fog ", "Sunny", "Cloudy", "Stormy"],
            "Road_traffic_density": ["Low", "Jam", "High", "Low", "Medium", "Jam"],
            "Type_of_order": ["Snack", "Meal", "Drinks", "Buffet", "Snack", "Meal"],
            "Type_of_vehicle": ["motorcycle"] * 6,
            "Festival": ["No"] * 6,
            "City": ["Urban"] * 6,
            "delivery_speed": ["Average", "Slow", "Slow", "Fast", "Average", "Slow"],
        }
    )


def test_clean_dataset_strips_whitespace(sample_df):
    cleaned = analysis.clean_dataset(sample_df)
    assert cleaned["Weather_conditions"].tolist()[:3] == ["Sunny", "Stormy", "Fog"]


def test_clean_dataset_nulls_out_invalid_values(sample_df):
    cleaned = analysis.clean_dataset(sample_df)

    # Zero distance should become NaN rather than being silently kept
    # or dropped from the frame entirely.
    assert pd.isna(cleaned.loc[5, "distance_km"])

    # An age of 150 and a rating of 9.9 are outside plausible ranges.
    assert pd.isna(cleaned.loc[2, "Delivery_person_Age"])
    assert pd.isna(cleaned.loc[2, "Delivery_person_Ratings"])


def test_clean_dataset_nulls_out_negative_time():
    df = pd.DataFrame(
        {
            "Time_taken (min)": [20, -5],
            "distance_km": [2.0, 3.0],
            "Road_traffic_density": ["Low", "Jam"],
            "Weather_conditions": ["Sunny", "Stormy"],
        }
    )
    cleaned = analysis.clean_dataset(df)
    assert pd.isna(cleaned.loc[1, "Time_taken_min"])
    assert cleaned.loc[0, "Time_taken_min"] == 20


def test_clean_dataset_preserves_row_count(sample_df):
    cleaned = analysis.clean_dataset(sample_df)
    assert len(cleaned) == len(sample_df)


def test_traffic_impact_orders_by_severity(sample_df):
    cleaned = analysis.clean_dataset(sample_df)
    result = analysis.traffic_impact(cleaned)

    # Jam should show up with a higher average delivery time than Low
    # in this sample, and every traffic level present should appear.
    assert set(result["Road_traffic_density"].astype(str)) == {"Low", "Medium", "High", "Jam"}

    jam_time = result.loc[result["Road_traffic_density"] == "Jam", "avg_delivery_time"].iloc[0]
    low_time = result.loc[result["Road_traffic_density"] == "Low", "avg_delivery_time"].iloc[0]
    assert jam_time > low_time


def test_distance_impact_returns_positive_correlation(sample_df):
    cleaned = analysis.clean_dataset(sample_df)
    _, correlation = analysis.distance_impact(cleaned, bin_width_km=5.0)

    # In our sample data, longer distances line up with longer times,
    # so the correlation should be positive.
    assert correlation > 0


def test_weather_traffic_combo_sorted_descending(sample_df):
    cleaned = analysis.clean_dataset(sample_df)
    result = analysis.weather_traffic_combo(cleaned)

    times = result["avg_delivery_time"].tolist()
    assert times == sorted(times, reverse=True)


def test_generate_insights_returns_readable_strings(sample_df):
    cleaned = analysis.clean_dataset(sample_df)
    traffic_df = analysis.traffic_impact(cleaned)
    _, correlation = analysis.distance_impact(cleaned)
    combo_df = analysis.weather_traffic_combo(cleaned)

    insights = analysis.generate_insights(traffic_df, correlation, combo_df)

    assert len(insights) == 3
    assert all(isinstance(point, str) and len(point) > 0 for point in insights)


def test_load_dataset_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        analysis.load_dataset("data/this_file_does_not_exist.csv")
