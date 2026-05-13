import matplotlib.pyplot as plt
from pathlib import Path


def main(output_path: Path | str | None = None):
    # Initialize dataset
    data = [
        {"n": 4,  "ext_lb": 500, "ext_ub": 500, "pha": 460},
        {"n": 6,  "ext_lb": 520, "ext_ub": 520, "pha": 570},
        {"n": 8,  "ext_lb": 450, "ext_ub": 530, "pha": 520},
        {"n": 10, "ext_lb": 470, "ext_ub": 590, "pha": 550},
        {"n": 12, "ext_lb": 210, "ext_ub": 820, "pha": 600},
    ]

    ns = [d["n"] for d in data]
    ext_lb = [d["ext_lb"] for d in data]
    ext_ub = [d["ext_ub"] for d in data]
    pha = [d["pha"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot PHA line
    ax.plot(ns, pha, marker="o", linestyle="-", color="#1f77b4", label="Objective PHA")

    # Plot vertical error bars (as lines with caps) indicating [ext_lb, ext_ub]
    for x, ylow, yhigh in zip(ns, ext_lb, ext_ub):
        ax.vlines(x, ylow, yhigh, color="#ff7f0e", linewidth=2)
        # end caps
        cap_width = 0.12
        ax.hlines(ylow, x - cap_width, x + cap_width, color="#ff7f0e", linewidth=2)
        ax.hlines(yhigh, x - cap_width, x + cap_width, color="#ff7f0e", linewidth=2)

    ax.set_xlabel("n")
    ax.set_ylabel("Objective value")
    ax.set_title("Extensive-form bounds vs PHA objective")
    ax.set_xticks(ns)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(output_path) if output_path is not None else out_dir / "extensive_vs_pha.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved plot to: {out_file}")


if __name__ == "__main__":
    main()
