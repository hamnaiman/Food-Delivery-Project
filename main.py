
import sys
from pathlib import Path

from dotenv import load_dotenv

from src import analysis
from src import visualization
from src import ai_explanation


load_dotenv()

DATA_PATH = "data/food_delivery_dataset.csv"
CHARTS_DIR = "outputs/charts"
REPORT_PATH = "outputs/report.txt"


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run() -> None:
    print_section("STEP 1: Loading and cleaning data")
    raw_df = analysis.load_dataset(DATA_PATH)
    df = analysis.clean_dataset(raw_df)
    print(f"Loaded {len(raw_df)} rows, cleaned dataset ready for analysis.")

    summary = analysis.basic_summary(df)
    print(f"Average delivery time: {summary['avg_delivery_time_min']} minutes")
    print(f"Average distance: {summary['avg_distance_km']} km")

    print_section("STEP 2: Answering the core questions")

    print("\n[Q1] Impact of road traffic density on delivery time:")
    traffic_df = analysis.traffic_impact(df)
    print(traffic_df.to_string(index=False))

    print("\n[Q2] Impact of distance on delivery time:")
    distance_df, distance_corr = analysis.distance_impact(df)
    print(distance_df.to_string(index=False))
    print(f"Correlation (distance vs delivery time): {distance_corr}")

    print("\n[Q3] Combined effect of weather and traffic:")
    combo_df = analysis.weather_traffic_combo(df)
    print(combo_df.head(10).to_string(index=False))

    print_section("STEP 3: Generating charts")
    Path(CHARTS_DIR).mkdir(parents=True, exist_ok=True)

    chart_paths = [
        visualization.plot_traffic_vs_time(traffic_df, CHARTS_DIR),
        visualization.plot_distance_vs_time(df, CHARTS_DIR),
        visualization.plot_weather_traffic_heatmap(combo_df, CHARTS_DIR),
        visualization.plot_delivery_speed_breakdown(df, CHARTS_DIR),
    ]
    for path in chart_paths:
        print(f"Saved chart: {path}")

    print_section("STEP 4: Business insights")
    insights = analysis.generate_insights(traffic_df, distance_corr, combo_df)
    for point in insights:
        print(f"- {point}")

    print_section("STEP 5: AI-generated explanation")
    explanation = ai_explanation.explain_findings(summary, traffic_df, distance_corr, combo_df, insights)
    print(explanation)

    # Persist a plain-text report so the findings survive after this run.
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("FOOD DELIVERY ANALYTICS - REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total orders analyzed: {summary['total_orders']}\n")
        f.write(f"Average delivery time: {summary['avg_delivery_time_min']} minutes\n")
        f.write(f"Average distance: {summary['avg_distance_km']} km\n\n")
        f.write("Traffic impact:\n")
        f.write(traffic_df.to_string(index=False) + "\n\n")
        f.write("Distance impact:\n")
        f.write(distance_df.to_string(index=False) + "\n")
        f.write(f"Correlation: {distance_corr}\n\n")
        f.write("Weather + traffic combinations (top 10):\n")
        f.write(combo_df.head(10).to_string(index=False) + "\n\n")
        f.write("Business insights:\n")
        for point in insights:
            f.write(f"- {point}\n")
        f.write("\nAI-generated explanation:\n")
        f.write(explanation + "\n")

    print(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
