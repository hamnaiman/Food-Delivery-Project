

from pathlib import Path
import matplotlib

# Use a non-interactive backend since this runs headless (no display),
# e.g. in CI pipelines or plain terminal execution.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CHART_STYLE = {
    "figure.figsize": (9, 5.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.edgecolor": "#444444",
    "font.size": 10,
}


def _ensure_output_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_traffic_vs_time(traffic_df: pd.DataFrame, output_dir: str) -> str:
    """
    Bar chart: average delivery time per traffic density level.

    A bar chart is the right tool here because traffic density is
    categorical (Low/Medium/High/Jam) rather than continuous, so we're
    comparing discrete groups against each other.
    """
    out_dir = _ensure_output_dir(output_dir)

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots()
        bars = ax.bar(
            traffic_df["Road_traffic_density"].astype(str),
            traffic_df["avg_delivery_time"],
            color="#3B6FA0",
        )

        # Label each bar with its value - saves the reader from having
        # to eyeball the y-axis for exact numbers.
        for bar, value in zip(bars, traffic_df["avg_delivery_time"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title("Average Delivery Time by Road Traffic Density")
        ax.set_xlabel("Road Traffic Density")
        ax.set_ylabel("Average Delivery Time (minutes)")
        fig.tight_layout()

        out_path = out_dir / "traffic_vs_delivery_time.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return str(out_path)


def plot_distance_vs_time(df: pd.DataFrame, output_dir: str) -> str:
    """
    Scatter plot: raw distance vs delivery time across all orders.

    Unlike the traffic chart, distance is continuous, so a scatter
    plot is more appropriate here - it shows the actual spread of
    the relationship rather than collapsing it into buckets.
    """
    out_dir = _ensure_output_dir(output_dir)
    clean = df.dropna(subset=["distance_km", "Time_taken_min"])

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots()
        ax.scatter(
            clean["distance_km"],
            clean["Time_taken_min"],
            alpha=0.15,
            s=10,
            color="#C0562F",
        )

        # Overlay a binned average line on top of the raw scatter - it
        # cuts through the noise and shows the underlying trend without
        # hiding how spread out individual deliveries actually are.
        clean["_distance_bin"] = pd.cut(clean["distance_km"], bins=20)
        binned_avg = clean.groupby("_distance_bin", observed=True).agg(
            mid=("distance_km", "mean"), avg_time=("Time_taken_min", "mean")
        )
        ax.plot(binned_avg["mid"], binned_avg["avg_time"], color="#1B1B1B", linewidth=2, label="Binned average")
        ax.legend()

        ax.set_title("Delivery Distance vs Delivery Time")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Delivery Time (minutes)")
        fig.tight_layout()

        out_path = out_dir / "distance_vs_delivery_time.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return str(out_path)


def plot_weather_traffic_heatmap(combo_df: pd.DataFrame, output_dir: str) -> str:
    """
    Heatmap-style grid: average delivery time for every weather x
    traffic combination.

    This is the chart that answers "what's the worst-case scenario"
    at a glance - darker cells mean slower deliveries.
    """
    out_dir = _ensure_output_dir(output_dir)

    pivot = combo_df.pivot(
        index="Weather_conditions", columns="Road_traffic_density", values="avg_delivery_time"
    )

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, cmap="OrRd", aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=0)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Annotate each cell with its actual number - a heatmap without
        # numbers forces the reader to guess, which defeats the point.
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9)

        ax.set_title("Average Delivery Time: Weather x Traffic")
        fig.colorbar(im, ax=ax, label="Avg delivery time (min)")
        fig.tight_layout()

        out_path = out_dir / "weather_traffic_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return str(out_path)


def plot_delivery_speed_breakdown(df: pd.DataFrame, output_dir: str) -> str:
    """
    Bar chart: how orders split across the Slow/Average/Fast delivery
    speed categories already present in the dataset. A nice quick
    sanity check alongside the three required questions.
    """
    out_dir = _ensure_output_dir(output_dir)
    counts = df["delivery_speed"].value_counts().reindex(["Slow", "Average", "Fast"]).dropna()

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color=["#B23A48", "#E8A33D", "#4E8B54"])
        ax.set_title("Order Volume by Delivery Speed Category")
        ax.set_xlabel("Delivery Speed")
        ax.set_ylabel("Number of Orders")
        fig.tight_layout()

        out_path = out_dir / "delivery_speed_breakdown.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return str(out_path)
