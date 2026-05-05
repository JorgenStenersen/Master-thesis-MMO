import matplotlib.pyplot as plt
import numpy as np

# Import all price lists from main.py
from visual_processing.main_dp import (
    prices_cm_up_t1, prices_cm_up_t2, prices_cm_up_t3, prices_cm_up_t4,
    prices_cm_down_t1, prices_cm_down_t2, prices_cm_down_t3, prices_cm_down_t4,
    prices_dam_t1, prices_dam_t2, prices_dam_t3, prices_dam_t4,
    prices_eam_up_t1, prices_eam_up_t2, prices_eam_up_t3, prices_eam_up_t4,
    prices_eam_down_t1, prices_eam_down_t2, prices_eam_down_t3, prices_eam_down_t4,
)

MARKETS = {
    "CM up": [
        prices_cm_up_t1,
        prices_cm_up_t2,
        prices_cm_up_t3,
        prices_cm_up_t4,
    ],
    "CM down": [
        prices_cm_down_t1,
        prices_cm_down_t2,
        prices_cm_down_t3,
        prices_cm_down_t4,
    ],
    "DAM": [
        prices_dam_t1,
        prices_dam_t2,
        prices_dam_t3,
        prices_dam_t4,
    ],
    "EAM up": [
        prices_eam_up_t1,
        prices_eam_up_t2,
        prices_eam_up_t3,
        prices_eam_up_t4,
    ],
    "EAM down": [
        prices_eam_down_t1,
        prices_eam_down_t2,
        prices_eam_down_t3,
        prices_eam_down_t4,
    ],
}

def _plot_market_axes(axes, markets, x_labels, positions, y_limits):
    for ax, (market, data) in zip(axes, markets.items()):
        ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(linewidth=1.4),
            boxprops=dict(linewidth=1.1),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
        )

        ax.set_title(market, fontsize=24)
        ax.set_xlabel("Timestamp", fontsize=20)
        ax.set_ylabel("Price (€/MWh)", fontsize=20)

        if y_limits:
            ax.set_ylim(*y_limits)

        ax.set_xlim(positions[0] - 0.4, positions[-1] + 0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.tick_params(labelsize=20)

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)


def plot_price_boxplots():
    x_labels = ["T1", "T2", "T3", "T4"]
    positions = np.arange(1, len(x_labels) + 1) * 0.8

    all_values = [v for series in MARKETS.values() for ts in series for v in ts]
    if all_values:
        y_low, y_high = np.percentile(all_values, [1, 99])
        padding = 0.1 * (y_high - y_low) if y_high > y_low else 1.0
        y_limits = (y_low - padding, y_high + padding)
    else:
        y_limits = None

    markets_four = {
        "CM up": MARKETS["CM up"],
        "CM down": MARKETS["CM down"],
        "EAM up": MARKETS["EAM up"],
        "EAM down": MARKETS["EAM down"],
    }

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 10),
        sharex=True,
        sharey=True
    )
    axes = axes.ravel()

    _plot_market_axes(axes, markets_four, x_labels, positions, y_limits)

    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.12,
        left=0.06,
        right=0.99,
        top=0.96,
        bottom=0.08
    )

    dam_market = {"DA": MARKETS["DAM"]}
    fig, ax = plt.subplots(figsize=(13, 5), sharex=True, sharey=True)
    _plot_market_axes([ax], dam_market, x_labels, positions, y_limits)
    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.12,
        left=0.06,
        right=0.99,
        top=0.96,
        bottom=0.12
    )

    plt.show()


if __name__ == "__main__":
    plot_price_boxplots()
