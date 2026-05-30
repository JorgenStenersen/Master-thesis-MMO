import matplotlib.pyplot as plt
from pathlib import Path


def plot_ph_boxplot(results, consensus, output_path: str | Path | None = None, *, var: str = "x", fontsize: int = 16, ylabel: str | None = None):
    """Create a boxplot of PH decisions with consensus diamonds.

    - results: list of per-bundle result dicts (may contain None entries)
    - consensus: consensus dict returned by PH (stage1, stage2, stage3)
    - output_path: optional path to save the figure (PNG). If None, the
      figure will be shown (but in non-interactive runs we save to a
      temporary file instead).

    The plot contains 5 boxes in this order: CM-up, CM-down, DA, EAM-up, EAM-down.
    For DA we sample decisions for a single parent u (the first found). For EAM
    we sample decisions for a single history v (the first found).
    """

    # Filter solved bundles
    solved = [r for r in results if r is not None]
    if not solved:
        raise ValueError("No solved bundle results available to plot.")

    # Helper to safely collect values (either 'x' or 'r')
    if var not in ("x", "r"):
        raise ValueError("var must be 'x' or 'r'")

    def collect_stage1(market):
        vals = []
        for r in solved:
            if market in r.get("stage1", {}):
                vals.append(r["stage1"][market][var])
        return vals

    def collect_stage2(market, u_key):
        vals = []
        for r in solved:
            v = r.get("stage2", {}).get((market, u_key))
            if v is not None:
                vals.append(v[var])
        return vals

    def collect_stage3(market, v_key):
        vals = []
        for r in solved:
            w = r.get("stage3", {}).get((market, v_key))
            if w is not None:
                vals.append(w[var])
        return vals

    # CM markets
    cm_up_vals = collect_stage1("CM_up")
    cm_down_vals = collect_stage1("CM_down")

    # DA: pick first (m,u) key in consensus stage2
    stage2_keys = list(consensus.get("stage2", {}).keys())
    if not stage2_keys:
        raise ValueError("No stage2 (DA) consensus entries available.")
    # find first key where market == 'DA'
    da_key = None
    for key in stage2_keys:
        if key[0] == "DA":
            da_key = key
            break
    if da_key is None:
        # fallback to first available
        da_key = stage2_keys[0]
    _, sample_u = da_key
    da_vals = collect_stage2("DA", sample_u)

    # EAM: pick first v
    stage3_keys = list(consensus.get("stage3", {}).keys())
    if not stage3_keys:
        raise ValueError("No stage3 (EAM) consensus entries available.")
    # extract unique v values and pick the first
    v_candidates = []
    for m, v in stage3_keys:
        if v not in v_candidates:
            v_candidates.append(v)
    sample_v = v_candidates[0]
    eam_up_vals = collect_stage3("EAM_up", sample_v)
    eam_down_vals = collect_stage3("EAM_down", sample_v)

    data = [cm_up_vals, cm_down_vals, da_vals, eam_up_vals, eam_down_vals]

    # Consensus markers (use var)
    cons_vals = []
    # CM
    cons_stage1 = consensus.get("stage1", {})
    cons_vals.append(cons_stage1.get("CM_up", {}).get(var, float("nan")))
    cons_vals.append(cons_stage1.get("CM_down", {}).get(var, float("nan")))
    # DA
    cons_stage2 = consensus.get("stage2", {})
    cons_vals.append(cons_stage2.get(("DA", sample_u), {}).get(var, float("nan")))
    # EAM
    cons_stage3 = consensus.get("stage3", {})
    cons_vals.append(cons_stage3.get(("EAM_up", sample_v), {}).get(var, float("nan")))
    cons_vals.append(cons_stage3.get(("EAM_down", sample_v), {}).get(var, float("nan")))

    # Plot
    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(figsize=(9, 6))
    box = ax.boxplot(data, patch_artist=True, widths=0.6, labels=["CM-up", "CM-down", "DA", "EAM-up", "EAM-down"]) 

    # Style boxes
    for patch in box.get("boxes", []):
        patch.set(facecolor="#1f77b4", edgecolor="black")

    # Draw consensus diamonds
    x_positions = [1, 2, 3, 4, 5]
    ax.scatter(x_positions, cons_vals, marker="D", color="red", s=80, zorder=3)

    if ylabel is None:
        if var == "x":
            ylabel = "Bid quantity decision (MW)"
        else:
            ylabel = "Bid price decision (€/MWh)"

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticklabels(["CM-up", "CM-down", "DA", "EAM-up", "EAM-down"], fontsize=fontsize)
    ax.yaxis.get_label().set_fontsize(fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    # No title
    plt.tight_layout()

    if output_path is None:
        # If no output path is given, return the figure object for the caller
        return fig, ax

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    return outp
