# plot_forecast_power_boxplots.py
# Boxplots (T1–T4) for forecasted power, pulled directly from main.py

import matplotlib.pyplot as plt

# ✅ Edit these names to match the variable names you have in main.py
from visual_processing.main_dp import (
    forecasted_power_t1,
    forecasted_power_t2,
    forecasted_power_t3,
    forecasted_power_t4,
)

def plot_forecast_power_boxplots():
    timestamps = ["T1", "T2", "T3", "T4"]
    data = [
        forecasted_power_t1,
        forecasted_power_t2,
        forecasted_power_t3,
        forecasted_power_t4,
    ]

    fig, ax = plt.subplots(figsize=(6, 11))

    ax.boxplot(
        data,
        widths=0.6,
        showfliers=True,          # set False if you want to hide outliers
        patch_artist=True,
        medianprops=dict(linewidth=1.4),
        boxprops=dict(linewidth=1.1),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
    )
    ax.set_title("Forecasted power by timestamp", fontsize=24)
    ax.set_xlabel("Timestamp", fontsize=20)
    ax.set_ylabel("Forecasted power (MW)", fontsize=20)
    ax.set_xticks(range(1, len(timestamps) + 1))
    ax.set_xticklabels(timestamps, fontsize=20)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_forecast_power_boxplots()
