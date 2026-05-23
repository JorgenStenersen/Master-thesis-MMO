import math
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


def main(output_path: Path | str | None = None):
    # Initialize dataset
    # T1
    
    data = [
        {"n": 4,  "ext_lb": 7177.2, "ext_ub": 7177.2, "pha": 7389.1},
        {"n": 5,  "ext_lb": 7277.8, "ext_ub": 7277.8, "pha": 6778.1},
        {"n": 6,  "ext_lb": 7062.5, "ext_ub": 7062.5, "pha": 6469.1},
        {"n": 7,  "ext_lb": 7194.5, "ext_ub": 7196.5, "pha": 8107.0},
        {"n": 8,  "ext_lb": 4767.7, "ext_ub": 7188.2, "pha": 6758.9},
        {"n": 9,  "ext_lb": 5930.2, "ext_ub": 7291.1, "pha": 5441.9},
        {"n": 10, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 5660.8},
        {"n": 11, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 6665.9},
        {"n": 12, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 6097.7},
    ]
    '''
    #---------T2---------
    data = [
    {"n": 4,  "ext_lb": 3530.0, "ext_ub": 3530.0, "pha": 3114.9},
    {"n": 5,  "ext_lb": 4093.9, "ext_ub": 4093.9, "pha": 5646.4},
    {"n": 6,  "ext_lb": 3646.0, "ext_ub": 3646.0, "pha": 3682.2},
    {"n": 7,  "ext_lb": 3607.6, "ext_ub": 3607.6, "pha": 4126.0},
    {"n": 8,  "ext_lb": 3564.7, "ext_ub": 3639.6, "pha": 4068.0},
    {"n": 9,  "ext_lb": 3542.7, "ext_ub": 3864.6, "pha": 3832.0},
    {"n": 10, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 3577.5},
    {"n": 11, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 3514.5},
    {"n": 12, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 4266.0},
    ]
    
    #---------T3---------
    data = [
    {"n": 4, "ext_lb": 812.7,  "ext_ub": 812.7,  "pha": 584.7},
    {"n": 5, "ext_lb": 914.5,  "ext_ub": 914.5,  "pha": 933.0},
    {"n": 6, "ext_lb": 950.7,  "ext_ub": 950.7,  "pha": 1014.0},
    {"n": 7, "ext_lb": 1004.5, "ext_ub": 1004.5, "pha": 965.6},
    {"n": 8, "ext_lb": 1007.5, "ext_ub": 1007.5, "pha": 919.0},
    {"n": 9, "ext_lb": 1164.4, "ext_ub": 1406.4, "pha": 1096.2},
    {"n": 10, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 998.2},
    {"n": 11, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 1029.6},
    {"n": 12, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 1119.2},
    ]
    
    
    #---------T4---------
    data = [
    {"n": 4,  "ext_lb": 3509.1, "ext_ub": 3509.1, "pha": 3602.0},
    {"n": 5,  "ext_lb": 3354.9, "ext_ub": 3354.9, "pha": 4392.6},
    {"n": 6,  "ext_lb": 3496.8, "ext_ub": 3572.8, "pha": 4631.7},
    {"n": 7,  "ext_lb": 2900.1, "ext_ub": 3938.5, "pha": 4957.0},
    {"n": 8,  "ext_lb": 2624.2, "ext_ub": 5365.0, "pha": 5259.3},
    {"n": 9,  "ext_lb": 3103.3, "ext_ub": 3292.7, "pha": 3785.5},
    {"n": 10, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 2713.8},
    {"n": 11, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 2726.0},
    {"n": 12, "ext_lb": float("nan"), "ext_ub": float("nan"), "pha": 4277.7},
    ]
    '''
    ns = [d["n"] for d in data]
    ext_lb = [d["ext_lb"] for d in data]
    ext_ub = [d["ext_ub"] for d in data]
    pha = [d["pha"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot PHA line
    pha_line = ax.plot(ns, pha, marker="o", linestyle="-", color="#1f77b4", label="Objective LPHA $T_1$")[0]

    # Plot vertical error bars (as lines with caps) indicating [ext_lb, ext_ub]
    for x, ylow, yhigh in zip(ns, ext_lb, ext_ub):
        ax.vlines(x, ylow, yhigh, color="#ff7f0e", linewidth=2)
        # end caps
        cap_width = 0.12
        ax.hlines(ylow, x - cap_width, x + cap_width, color="#ff7f0e", linewidth=2)
        ax.hlines(yhigh, x - cap_width, x + cap_width, color="#ff7f0e", linewidth=2)

    ax.set_xlabel("n", fontsize=18)
    ax.set_ylabel("Objective value", fontsize=18)
    #ax.set_title("Extensive-form bounds vs PHA objective")
    ax.set_xticks(ns)
    ax.tick_params(axis="both", labelsize=14)
    y_max = max(v for v in ext_ub + pha if not math.isnan(v))
    ax.set_ylim(0, y_max+300)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    esm_handle = Line2D([0], [0], color="#ff7f0e", lw=2)
    ax.legend(handles=[pha_line, esm_handle], labels=[pha_line.get_label(), "Objective ESM $T_1$"], fontsize=14, loc="lower right")

    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(output_path) if output_path is not None else out_dir / "extensive_vs_pha_T1.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved plot to: {out_file}")


if __name__ == "__main__":
    main()
