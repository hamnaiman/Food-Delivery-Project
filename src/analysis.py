

from pathlib import Path
import pandas as pd
import numpy as np


# Order matters here - it reflects how congested each level actually is,
# not just alphabetical order, which is what we need for any "low to high"
# style comparisons later on.
TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Read the raw CSV off disk and return it as a DataFrame.

    We keep this as its own function (rather than inlining pd.read_csv
    everywhere) so that if the data source ever changes - a different
    delimiter, a database instead of a file, whatever - there's exactly
    one place to update.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at '{csv_path}'")

    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce an analysis-ready copy of the raw dataframe.

    The raw file has a few rough edges that are typical of real-world
    delivery data:
      - stray whitespace around categorical text (e.g. " Jam" vs "Jam")
      - missing ages, ratings and order times
      - a couple of column names with awkward spacing/casing

    None of these are fatal, but they will quietly break groupby()
    comparisons if left alone, so we handle them up front.
    """
    data = df.copy()

    # Tidy up column names we know we'll reference a lot, without
    # renaming everything blindly (some downstream code still expects
    # the original names).
    data = data.rename(columns={"Time_taken (min)": "Time_taken_min"})

    # Strip whitespace from the categorical columns that drive our
    # groupings. Trailing spaces are a classic source of "why are there
    # two Jam categories" bugs.
    categorical_cols = [
        "Weather_conditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
        "delivery_speed",
    ]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
            # Some source files use "NaN" as a literal string once
            # cast through astype(str) - convert those back to real NaN.
            data.loc[data[col].isin(["nan", "NaN", ""]), col] = np.nan

    # Numeric columns that should never be negative or absurdly large.
    # We don't drop these rows outright (that would waste useful data
    # points where only one field is odd) - instead we null out the
    # offending value so it's excluded from averages but the rest of
    # the row survives.
    if "distance_km" in data.columns:
        data.loc[data["distance_km"] <= 0, "distance_km"] = np.nan

    if "Time_taken_min" in data.columns:
        data.loc[data["Time_taken_min"] <= 0, "Time_taken_min"] = np.nan

    if "Delivery_person_Age" in data.columns:
        data.loc[
            (data["Delivery_person_Age"] < 15) | (data["Delivery_person_Age"] > 80),
            "Delivery_person_Age",
        ] = np.nan

    if "Delivery_person_Ratings" in data.columns:
        data.loc[
            (data["Delivery_person_Ratings"] < 1) | (data["Delivery_person_Ratings"] > 5),
            "Delivery_person_Ratings",
        ] = np.nan

    # Road_traffic_density is naturally ordered (Low < Medium < High < Jam).
    # Encoding it as an ordered category lets pandas sort groupby output
    # sensibly instead of alphabetically, which is nicer for charts too.
    if "Road_traffic_density" in data.columns:
        present_levels = [lvl for lvl in TRAFFIC_ORDER if lvl in data["Road_traffic_density"].unique()]
        data["Road_traffic_density"] = pd.Categorical(
            data["Road_traffic_density"], categories=present_levels, ordered=True
        )

    return data


def basic_summary(df: pd.DataFrame) -> dict:
    """
    A quick set of headline numbers used at the top of the report -
    row count, missing data, and simple central tendencies for the
    fields people usually care about first.
    """
    return {
        "total_orders": int(len(df)),
        "missing_values": df.isnull().sum().to_dict(),
        "avg_delivery_time_min": round(df["Time_taken_min"].mean(), 2),
        "median_delivery_time_min": round(df["Time_taken_min"].median(), 2),
        "avg_distance_km": round(df["distance_km"].mean(), 2),
        "avg_delivery_person_rating": round(df["Delivery_person_Ratings"].mean(), 2),
    }


def traffic_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 1: How does road traffic density affect delivery time?

    Groups orders by traffic level and reports average/median delivery
    time plus order volume, so we can see both "how much slower" and
    "how common is this situation" in one table.
    """
    result = (
        df.dropna(subset=["Road_traffic_density", "Time_taken_min"])
        .groupby("Road_traffic_density", observed=True)["Time_taken_min"]
        .agg(avg_delivery_time="mean", median_delivery_time="median", order_count="count")
        .round(2)
        .reset_index()
    )
    return result


def distance_impact(df: pd.DataFrame, bin_width_km: float = 5.0) -> pd.DataFrame:
    """
    Question 2: How does delivery distance affect delivery time?

    Rather than just computing a single correlation coefficient (which
    hides a lot of nuance), we bucket distance into bands and show the
    average delivery time per band. This makes it easy to spot, for
    example, whether time grows roughly linearly with distance or
    plateaus after a point.
    """
    clean = df.dropna(subset=["distance_km", "Time_taken_min"]).copy()

    max_distance = clean["distance_km"].max()
    bin_edges = np.arange(0, max_distance + bin_width_km, bin_width_km)
    clean["distance_band"] = pd.cut(clean["distance_km"], bins=bin_edges)

    result = (
        clean.groupby("distance_band", observed=True)["Time_taken_min"]
        .agg(avg_delivery_time="mean", order_count="count")
        .round(2)
        .reset_index()
    )
    result["distance_band"] = result["distance_band"].astype(str)

    correlation = clean["distance_km"].corr(clean["Time_taken_min"])

    return result, round(correlation, 3)


def weather_traffic_combo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 3: What happens when bad weather and heavy traffic combine?

    Cross-tabulates weather against traffic density and reports the
    average delivery time for each combination, so we can identify the
    genuinely worst-case pairings (e.g. Stormy + Jam) rather than just
    looking at each factor in isolation.
    """
    clean = df.dropna(subset=["Weather_conditions", "Road_traffic_density", "Time_taken_min"])

    result = (
        clean.groupby(["Weather_conditions", "Road_traffic_density"], observed=True)["Time_taken_min"]
        .agg(avg_delivery_time="mean", order_count="count")
        .round(2)
        .reset_index()
        .sort_values("avg_delivery_time", ascending=False)
    )
    return result


def generate_insights(traffic_df: pd.DataFrame, distance_corr: float, combo_df: pd.DataFrame) -> list:
    """
    Turn the numeric results above into short, plain-language business
    insights. These are computed from whatever data was passed in - no
    figures are hardcoded, so the insights stay accurate even if the
    underlying dataset changes.
    """
    insights = []

    if not traffic_df.empty:
        slowest = traffic_df.loc[traffic_df["avg_delivery_time"].idxmax()]
        fastest = traffic_df.loc[traffic_df["avg_delivery_time"].idxmin()]
        gap = round(slowest["avg_delivery_time"] - fastest["avg_delivery_time"], 1)
        insights.append(
            f"Deliveries made during '{slowest['Road_traffic_density']}' traffic take about "
            f"{gap} minutes longer on average than during '{fastest['Road_traffic_density']}' "
            f"traffic - traffic density is one of the biggest levers on delivery time."
        )

    if distance_corr is not None:
        strength = "strong" if abs(distance_corr) > 0.5 else "moderate" if abs(distance_corr) > 0.2 else "weak"
        insights.append(
            f"Distance and delivery time show a {strength} positive relationship "
            f"(correlation = {distance_corr}), confirming that longer routes reliably take more time."
        )

    if not combo_df.empty:
        worst = combo_df.iloc[0]
        insights.append(
            f"The slowest combination overall is '{worst['Weather_conditions']}' weather with "
            f"'{worst['Road_traffic_density']}' traffic, averaging {worst['avg_delivery_time']} "
            f"minutes per delivery - this is the scenario most worth planning extra buffer time for."
        )

    return insights
