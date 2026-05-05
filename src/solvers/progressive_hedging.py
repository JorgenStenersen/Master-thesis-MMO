import json
import math
from datetime import datetime, timezone
from pathlib import Path

import src.read as read

import gurobipy as gp
from gurobipy import GRB

import src.utils as utils
from src.model import build_model, get_market_products
import src.tree as tree
from src.read import get_global_bounds_from_raw_data


def build_bundle_models(B, global_bounds, verbose=False, gurobi_threads=None):
    """
    Pre-builds a Gurobi model for each bundle. These models are reused
    across all PH iterations to avoid the overhead of re-creating
    variables, constraints, and index sets every time.

    Input:
        B:              list of scenario tree dicts
        global_bounds:  dict with global Big-M bounds
        verbose:        if True, print progress

    Returns:
        models:      list of ModelContainer objects (one per bundle)
        base_objs:   list of base objective expressions (without PH penalties)
    """
    models = []
    base_objs = []

    for b_idx, bundle_tree in enumerate(B):
        mc = build_model(bundle_tree, global_bounds, mode="progressive_hedging")
        mc.model.setParam("OutputFlag", 0)
        if gurobi_threads is not None:
            mc.model.setParam("Threads", int(gurobi_threads))
        mc.model.update()
        base_objs.append(mc.model.getObjective())
        models.append(mc)

        if verbose:
            print(f"[INFO] Built model for bundle {b_idx}")

    return models, base_objs


def solve_bundles(B, global_bounds, market_products, models=None, base_objs=None, verbose=False, gurobi_threads=None):
    """
    Solves the model for each scenario tree (bundle) in B and stores
    the first-, second-, and third-stage decision variables.

    Input:
        B:              list of scenario tree dicts (output from build_scenario_bundles)
        global_bounds:  dict with global Big-M bounds (from get_global_bounds_from_input_data)
        market_products: tuple (M_u, M_v, M_w, M) from get_market_products()
        verbose:        if True, print per-bundle summaries

    Returns:
        results: list of dicts, one per bundle, each containing:
            - "objective"   : optimal objective value (negated back to original maximization sense)
            - "objective_unaugmented": unaugmented objective value (maximization sense)
            - "stage1"      : dict  {m: {"x": val, "r": val}}  for m in M_u  (CM markets)
                              (one representative value per market; NA enforced within each bundle)
            - "stage2"      : dict  {(m, u): {"x": val, "r": val}}  for m in M_v, u in U
                              (DA market, one representative per parent CM-node u)
            - "stage3"      : dict  {(m, v): {"x": val, "r": val}}  for m in M_w, v in V_all
                              (EAM markets, one representative per parent DA-node v)
    """

    M_u, M_v, M_w, M = market_products
    results = []

    for b_idx, bundle_tree in enumerate(B):

        # Use pre-built model if available, otherwise build from scratch
        if models is not None:
            mc = models[b_idx]
            # Reset objective to base (in case it was modified by a previous augmented solve)
            mc.model.setObjective(base_objs[b_idx], GRB.MINIMIZE)
        else:
            mc = build_model(bundle_tree, global_bounds, mode="progressive_hedging")
            mc.model.setParam("OutputFlag", 0)
        if gurobi_threads is not None:
            mc.model.setParam("Threads", int(gurobi_threads))
        mc.model.optimize()

        if mc.model.Status != GRB.OPTIMAL:
            print(f"[WARNING] Bundle {b_idx} not solved to optimality (status={mc.model.Status})")
            results.append(None)
            continue

        # Retrieve variable dicts
        x = mc.vars["x"]
        r = mc.vars["r"]

        # Retrieve sets
        U = mc.sets["U"]
        V = mc.sets["V"]
        W = mc.sets["W"]
        V_all = set().union(*V.values())

        # --- Stage 1 (first-stage decisions): CM markets ---
        # NA constraints make all u-nodes share the same value; pick any representative.
        u_rep = next(iter(U))
        stage1 = {}
        for m in M_u:
            stage1[m] = {
                "x": x[m, u_rep].X,
                "r": r[m, u_rep].X,
            }

        # --- Stage 2 (second-stage decisions): DA market ---
        # Within children of a given u, NA enforces the same value.
        # Store one representative per parent u.
        stage2 = {}
        for u in U:
            v_rep = next(iter(V[u]))  # representative v-node for this u
            for m in M_v:
                stage2[(m, u)] = {
                    "x": x[m, v_rep].X,
                    "r": r[m, v_rep].X,
                }

        # --- Stage 3 (third-stage decisions): EAM markets ---
        # Within children of a given v, NA enforces the same value.
        # Store one representative per parent v.
        stage3 = {}
        for v in V_all:
            w_rep = next(iter(W[v]))  # representative w-node for this v
            for m in M_w:
                stage3[(m, v)] = {
                    "x": x[m, w_rep].X,
                    "r": r[m, w_rep].X,
                }

        # Negate objective back to maximization sense
        obj_val = -mc.model.ObjVal

        bundle_result = {
            "objective": obj_val,
            "objective_unaugmented": obj_val,
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }
        results.append(bundle_result)

        if verbose:
            print(f"[INFO] Bundle {b_idx}: obj = {obj_val:.4f}")

    return results


def compute_consensus(results, verbose=False):
    """
    Computes the consensus (weighted average) of decisions across all bundles
    for each node.  This corresponds to steps 6-8 of the PH algorithm:

        x̄_n = Σ_{b∈B} π_nb · x̂_nb

    where π_nb = 1 / |{bundles containing node n}|  (equal weight).

    For stage 1 (root): every bundle contributes → weight = 1/num_bundles.
    For stage 2 (u-nodes): only bundles that contain a given u-node contribute.
    For stage 3 (v-nodes): only bundles that contain a given v-node contribute.

    Input:
        results: list of per-bundle result dicts (from solve_bundles)
        verbose: if True, print summary

    Returns:
        consensus: dict with keys "stage1", "stage2", "stage3", each containing
                   the averaged decision values with the same key structure as
                   the individual bundle results.
    """

    # Filter out None results (unsolved bundles)
    solved = [r for r in results if r is not None]

    if not solved:
        raise ValueError("No bundles were solved successfully.")

    # ------------------------------------------------------------------
    # Stage 1 consensus  (root node — all bundles share it)
    # ------------------------------------------------------------------
    stage1_consensus = {}
    markets_stage1 = solved[0]["stage1"].keys()
    for m in markets_stage1:
        avg_x = sum(r["stage1"][m]["x"] for r in solved) / len(solved)
        avg_r = sum(r["stage1"][m]["r"] for r in solved) / len(solved)
        stage1_consensus[m] = {"x": avg_x, "r": avg_r}

    # ------------------------------------------------------------------
    # Stage 2 consensus  (u-nodes — may differ across bundles)
    # ------------------------------------------------------------------
    # For each (market, u-node) key, collect values from all bundles that
    # contain it, then average.
    stage2_accum = {}          # {(m, u): [{"x": ..., "r": ...}, ...]}
    for r in solved:
        for key, vals in r["stage2"].items():
            stage2_accum.setdefault(key, []).append(vals)

    stage2_consensus = {}
    for key, vals_list in stage2_accum.items():
        n_b = len(vals_list)
        avg_x = sum(v["x"] for v in vals_list) / n_b
        avg_r = sum(v["r"] for v in vals_list) / n_b
        stage2_consensus[key] = {"x": avg_x, "r": avg_r}

    # ------------------------------------------------------------------
    # Stage 3 consensus  (v-nodes — may differ across bundles)
    # ------------------------------------------------------------------
    stage3_accum = {}
    for r in solved:
        for key, vals in r["stage3"].items():
            stage3_accum.setdefault(key, []).append(vals)

    stage3_consensus = {}
    for key, vals_list in stage3_accum.items():
        n_b = len(vals_list)
        avg_x = sum(v["x"] for v in vals_list) / n_b
        avg_r = sum(v["r"] for v in vals_list) / n_b
        stage3_consensus[key] = {"x": avg_x, "r": avg_r}

    consensus = {
        "stage1": stage1_consensus,
        "stage2": stage2_consensus,
        "stage3": stage3_consensus,
    }

    if verbose:
        print(f"\n[CONSENSUS] Stage 1: {len(stage1_consensus)} market(s), "
              f"averaged over {len(solved)} bundles")
        print(f"[CONSENSUS] Stage 2: {len(stage2_consensus)} unique (market, u-node) pairs")
        print(f"[CONSENSUS] Stage 3: {len(stage3_consensus)} unique (market, v-node) pairs")

    return consensus


def initialize_shadow_costs(results, consensus, alpha=100.0, verbose=False):
    """
    Initialises the shadow costs (dual multipliers) for every node-bundle
    pair, corresponding to steps 9-10 of the PH algorithm:

        w_nb^(0) = alpha * (x_nb^(0) - x_bar_n^(0))

    Input:
        results:    list of per-bundle result dicts (from solve_bundles)
        consensus:  dict with "stage1", "stage2", "stage3" consensus values
        alpha:      penalty parameter (default 100)
        verbose:    if True, print summary

    Returns:
        W_shadow: list (one entry per bundle) of dicts, each with keys
                  "stage1", "stage2", "stage3".  Value structure mirrors
                  the results/consensus dicts:
                    stage1[m]      = {"x": w_val, "r": w_val}
                    stage2[(m, u)] = {"x": w_val, "r": w_val}
                    stage3[(m, v)] = {"x": w_val, "r": w_val}
    """

    W_shadow = []

    for b_idx, res in enumerate(results):
        if res is None:
            W_shadow.append(None)
            continue

        bundle_w = {"stage1": {}, "stage2": {}, "stage3": {}}

        # Stage 1
        for m, vals in res["stage1"].items():
            cons = consensus["stage1"][m]
            bundle_w["stage1"][m] = {
                "x": alpha * (vals["x"] - cons["x"]),
                "r": alpha * (vals["r"] - cons["r"]),
            }

        # Stage 2
        for key, vals in res["stage2"].items():
            cons = consensus["stage2"][key]
            bundle_w["stage2"][key] = {
                "x": alpha * (vals["x"] - cons["x"]),
                "r": alpha * (vals["r"] - cons["r"]),
            }

        # Stage 3
        for key, vals in res["stage3"].items():
            cons = consensus["stage3"][key]
            bundle_w["stage3"][key] = {
                "x": alpha * (vals["x"] - cons["x"]),
                "r": alpha * (vals["r"] - cons["r"]),
            }

        W_shadow.append(bundle_w)

    if verbose:
        n_init = sum(1 for w in W_shadow if w is not None)
        print(f"[SHADOW] Initialised shadow costs for {n_init} bundles (alpha={alpha})")

    return W_shadow


def compute_convergence_gap(results, consensus, market_products):
    """
    Computes the convergence gap (step 24 / while-condition 12):

        g^(k) = sum_{n in N} sum_{b in B} pi_nb * ||x_nb^(k) - xbar_n^(k)||

    where pi_nb = 1 / |{bundles containing node n}|.

    Input:
        results:         list of per-bundle result dicts
        consensus:       dict with "stage1", "stage2", "stage3" consensus values
        market_products: tuple (M_u, M_v, M_w, M) from get_market_products()

    Returns:
        gap: float, the total weighted distance from consensus
    """

    solved = [(i, r) for i, r in enumerate(results) if r is not None]
    if not solved:
        return float('inf')

    M_u, M_v, M_w, _ = market_products
    gap = 0.0
    num_solved = len(solved)

    # Stage 1 (root): all bundles contribute, pi = 1/num_solved
    pi = 1.0 / num_solved
    for _, res in solved:
        sq = sum(
            (res["stage1"][m]["x"] - consensus["stage1"][m]["x"])**2 +
            (res["stage1"][m]["r"] - consensus["stage1"][m]["r"])**2
            for m in M_u
        )
        gap += pi * math.sqrt(sq)

    # Stage 2 (u-nodes): only bundles containing u contribute
    u_to_bundles = {}
    for idx, res in solved:
        for (m, u) in res["stage2"]:
            u_to_bundles.setdefault(u, set()).add(idx)

    for u, bundle_set in u_to_bundles.items():
        pi_u = 1.0 / len(bundle_set)
        for idx in bundle_set:
            res = results[idx]
            sq = sum(
                (res["stage2"][(m, u)]["x"] - consensus["stage2"][(m, u)]["x"])**2 +
                (res["stage2"][(m, u)]["r"] - consensus["stage2"][(m, u)]["r"])**2
                for m in M_v
            )
            gap += pi_u * math.sqrt(sq)

    # Stage 3 (v-nodes): only bundles containing v contribute
    v_to_bundles = {}
    for idx, res in solved:
        for (m, v) in res["stage3"]:
            v_to_bundles.setdefault(v, set()).add(idx)

    for v, bundle_set in v_to_bundles.items():
        pi_v = 1.0 / len(bundle_set)
        for idx in bundle_set:
            res = results[idx]
            sq = sum(
                (res["stage3"][(m, v)]["x"] - consensus["stage3"][(m, v)]["x"])**2 +
                (res["stage3"][(m, v)]["r"] - consensus["stage3"][(m, v)]["r"])**2
                for m in M_w
            )
            gap += pi_v * math.sqrt(sq)

    return gap


def compute_dual_residual(consensus, prev_consensus, alpha):
    """
    Computes the dual residual, measuring how much the consensus changed
    between two successive PH iterations:

        s^(k) = alpha * || xbar^(k) - xbar^(k-1) ||

    Input:
        consensus:       current consensus dict
        prev_consensus:  consensus dict from the previous iteration
        alpha:           current penalty parameter

    Returns:
        dual_res: float, the dual residual
    """
    sq = 0.0

    # Stage 1
    for m in consensus["stage1"]:
        sq += (consensus["stage1"][m]["x"] - prev_consensus["stage1"][m]["x"])**2
        sq += (consensus["stage1"][m]["r"] - prev_consensus["stage1"][m]["r"])**2

    # Stage 2
    for key in consensus["stage2"]:
        sq += (consensus["stage2"][key]["x"] - prev_consensus["stage2"][key]["x"])**2
        sq += (consensus["stage2"][key]["r"] - prev_consensus["stage2"][key]["r"])**2

    # Stage 3
    for key in consensus["stage3"]:
        sq += (consensus["stage3"][key]["x"] - prev_consensus["stage3"][key]["x"])**2
        sq += (consensus["stage3"][key]["r"] - prev_consensus["stage3"][key]["r"])**2

    return alpha * math.sqrt(sq)


def adapt_alpha(alpha, primal_residual, dual_residual, tau=2.0, mu=10.0):
    """
    Residual-balancing adaptive alpha update (Boyd et al., ADMM survey).

    If the primal residual (convergence gap) is much larger than the dual
    residual, alpha is increased to push bundles harder toward consensus.
    If the dual residual dominates, alpha is decreased to avoid oscillation.

    Input:
        alpha:            current penalty parameter
        primal_residual:  convergence gap g^(k)
        dual_residual:    dual residual s^(k)
        tau:              scaling factor for alpha adjustments (default 2.0)
        mu:               threshold ratio for triggering adjustment (default 10.0)

    Returns:
        alpha: updated penalty parameter
    """
    if primal_residual > mu * dual_residual:
        alpha *= tau
    elif dual_residual > mu * primal_residual:
        alpha /= tau
    return alpha


def update_shadow_costs(W_shadow, results, consensus, alpha):
    """
    Updates shadow costs in-place, corresponding to steps 21-22:

        w_nb^(k) = w_nb^(k-1) + alpha * (x_nb^(k) - xbar_n^(k))

    Input:
        W_shadow:   list of shadow-cost dicts (one per bundle), modified in-place
        results:    list of per-bundle result dicts (updated decisions)
        consensus:  dict with updated consensus values
        alpha:      penalty parameter

    Returns:
        W_shadow (same list, updated in-place)
    """

    for b_idx, res in enumerate(results):
        if res is None or W_shadow[b_idx] is None:
            continue

        w_b = W_shadow[b_idx]

        # Stage 1
        for m in res["stage1"]:
            cons = consensus["stage1"][m]
            w_b["stage1"][m]["x"] += alpha * (res["stage1"][m]["x"] - cons["x"])
            w_b["stage1"][m]["r"] += alpha * (res["stage1"][m]["r"] - cons["r"])

        # Stage 2
        for key in res["stage2"]:
            cons = consensus["stage2"][key]
            w_b["stage2"][key]["x"] += alpha * (res["stage2"][key]["x"] - cons["x"])
            w_b["stage2"][key]["r"] += alpha * (res["stage2"][key]["r"] - cons["r"])

        # Stage 3
        for key in res["stage3"]:
            cons = consensus["stage3"][key]
            w_b["stage3"][key]["x"] += alpha * (res["stage3"][key]["x"] - cons["x"])
            w_b["stage3"][key]["r"] += alpha * (res["stage3"][key]["r"] - cons["r"])

    return W_shadow


def solve_bundles_augmented(B, global_bounds, W_shadow, consensus, alpha,
                           market_products, models=None, base_objs=None, verbose=False,
                           gurobi_threads=None):
    """
    Solves each bundle with the augmented PH objective (steps 14-16):

        min  -f(b) + sum_{n in N} [ w_nb^(k-1) * x_nb
                                     + alpha * ||x_nb - xbar_n^(k-1)||^2 ]

    The linear shadow-cost term steers decisions toward consensus.
    The quadratic proximity term penalises deviation from the previous
    consensus.

    If pre-built models are provided, they are reused (only the objective
    is updated) and the previous solution is used as a warm-start.

    Input:
        B:              list of scenario tree dicts
        global_bounds:  dict with global Big-M bounds
        W_shadow:       list of shadow-cost dicts (one per bundle)
        consensus:      current consensus dict
        alpha:          penalty parameter
        market_products: tuple (M_u, M_v, M_w, M) from get_market_products()
        models:         optional list of pre-built ModelContainer objects
        base_objs:      optional list of base objective expressions
        verbose:        if True, print per-bundle summaries

    Returns:
        results: list of per-bundle result dicts (same format as solve_bundles)
    """

    M_u, M_v, M_w, M = market_products
    results = []

    for b_idx, bundle_tree in enumerate(B):
        if W_shadow[b_idx] is None:
            results.append(None)
            continue

        # Use pre-built model if available, otherwise build from scratch
        if models is not None:
            mc = models[b_idx]
            # Warm-start: set Start attributes from previous solution
            try:
                for var in mc.model.getVars():
                    var.Start = var.X
            except AttributeError:
                pass  # No previous solution yet (first augmented iteration)
        else:
            mc = build_model(bundle_tree, global_bounds, mode="progressive_hedging")

        if gurobi_threads is not None:
            mc.model.setParam("Threads", int(gurobi_threads))

        x = mc.vars["x"]
        r = mc.vars["r"]
        U = mc.sets["U"]
        V = mc.sets["V"]
        W = mc.sets["W"]
        V_all = set().union(*V.values())

        w_b = W_shadow[b_idx]

        # ---- Build penalty expression ----
        penalty = gp.QuadExpr()

        # Stage 1 penalties (root node)
        u_rep = next(iter(U))
        for m in M_u:
            w_x = w_b["stage1"][m]["x"]
            w_r = w_b["stage1"][m]["r"]
            xbar_x = consensus["stage1"][m]["x"]
            xbar_r = consensus["stage1"][m]["r"]

            penalty += w_x * x[m, u_rep] + w_r * r[m, u_rep]
            penalty += alpha * (x[m, u_rep] - xbar_x) * (x[m, u_rep] - xbar_x)
            penalty += alpha * (r[m, u_rep] - xbar_r) * (r[m, u_rep] - xbar_r)

        # Stage 2 penalties (u-nodes)
        for u in U:
            v_rep = next(iter(V[u]))
            for m in M_v:
                key = (m, u)
                w_x = w_b["stage2"][key]["x"]
                w_r = w_b["stage2"][key]["r"]
                xbar_x = consensus["stage2"][key]["x"]
                xbar_r = consensus["stage2"][key]["r"]

                penalty += w_x * x[m, v_rep] + w_r * r[m, v_rep]
                penalty += alpha * (x[m, v_rep] - xbar_x) * (x[m, v_rep] - xbar_x)
                penalty += alpha * (r[m, v_rep] - xbar_r) * (r[m, v_rep] - xbar_r)

        # Stage 3 penalties (v-nodes)
        for v in V_all:
            w_rep = next(iter(W[v]))
            for m in M_w:
                key = (m, v)
                w_x = w_b["stage3"][key]["x"]
                w_r = w_b["stage3"][key]["r"]
                xbar_x = consensus["stage3"][key]["x"]
                xbar_r = consensus["stage3"][key]["r"]

                penalty += w_x * x[m, w_rep] + w_r * r[m, w_rep]
                penalty += alpha * (x[m, w_rep] - xbar_x) * (x[m, w_rep] - xbar_x)
                penalty += alpha * (r[m, w_rep] - xbar_r) * (r[m, w_rep] - xbar_r)

        # Add penalty to the base objective
        if models is not None:
            base_obj_expr = base_objs[b_idx]
            mc.model.setObjective(base_obj_expr + penalty, GRB.MINIMIZE)
        else:
            mc.model.update()
            base_obj_expr = mc.model.getObjective()
            mc.model.setObjective(base_obj_expr + penalty, GRB.MINIMIZE)

        mc.model.setParam("OutputFlag", 0)
        mc.model.optimize()

        if mc.model.Status != GRB.OPTIMAL:
            print(f"[WARNING] Bundle {b_idx} not solved to optimality (status={mc.model.Status})")
            results.append(None)
            continue

        # Extract decisions (same structure as solve_bundles)
        u_rep = next(iter(U))
        stage1 = {}
        for m in M_u:
            stage1[m] = {"x": x[m, u_rep].X, "r": r[m, u_rep].X}

        stage2 = {}
        for u in U:
            v_rep = next(iter(V[u]))
            for m in M_v:
                stage2[(m, u)] = {"x": x[m, v_rep].X, "r": r[m, v_rep].X}

        stage3 = {}
        for v in V_all:
            w_rep = next(iter(W[v]))
            for m in M_w:
                stage3[(m, v)] = {"x": x[m, w_rep].X, "r": r[m, w_rep].X}

        unaug_obj_val = None
        if base_obj_expr is not None:
            unaug_obj_val = -base_obj_expr.getValue()

        bundle_result = {
            "objective": -mc.model.ObjVal,
            "objective_unaugmented": unaug_obj_val,
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }
        results.append(bundle_result)

        if verbose:
            print(f"[INFO] Bundle {b_idx}: augmented obj = {mc.model.ObjVal:.4f}")

    return results


def _objective_mean(results, key="objective"):
    objectives = [r.get(key) for r in results if r is not None and r.get(key) is not None]
    if not objectives:
        return None
    return sum(objectives) / len(objectives)


def _gap_threshold_from_objective(objective_mean, gap_pct, epsilon):
    if objective_mean is None:
        return epsilon
    if not math.isfinite(objective_mean):
        return epsilon
    scaled = gap_pct * abs(objective_mean)
    if scaled <= 0.0:
        return epsilon
    return scaled


def _sanitize_time_str(time_str: str) -> str:
    safe = []
    for ch in time_str:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def format_final_consensus(consensus) -> str:
    """Return a formatted consensus summary string."""
    lines = []
    lines.append("")
    lines.append("=" * 82)
    lines.append("  CONVERGED CONSENSUS DECISIONS")
    lines.append("-" * 82)
    for stage_name, label in [("stage1", "CM  (stage 1)"),
                               ("stage2", "DA  (stage 2)"),
                               ("stage3", "EAM (stage 3)")]:
        stage = consensus[stage_name]
        if not stage:
            continue
        lines.append("")
        lines.append(f"  {label}  ({len(stage)} decision(s))")
        lines.append(f"  {'Key':>30s}  {'x':>10s}  {'r':>10s}")
        lines.append(f"  {'':->30s}  {'':->10s}  {'':->10s}")
        for key, vals in stage.items():
            lines.append(f"  {str(key):>30s}  {vals['x']:>10.4f}  {vals['r']:>10.4f}")
    lines.append("")
    lines.append("=" * 82)
    lines.append("")
    return "\n".join(lines)


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _tuple_sort_key(key):
    if isinstance(key, tuple) and len(key) >= 2:
        return (str(key[0]), str(key[1]))
    return (str(key), "")


def consensus_to_json(consensus):
    stage1_entries = []
    for m in sorted(consensus.get("stage1", {}).keys(), key=lambda k: str(k)):
        vals = consensus["stage1"][m]
        stage1_entries.append({
            "m": _json_safe(m),
            "x": float(vals["x"]),
            "r": float(vals["r"]),
        })

    stage2_entries = []
    for key in sorted(consensus.get("stage2", {}).keys(), key=_tuple_sort_key):
        m, u = key
        vals = consensus["stage2"][key]
        stage2_entries.append({
            "m": _json_safe(m),
            "u": _json_safe(u),
            "x": float(vals["x"]),
            "r": float(vals["r"]),
        })

    stage3_entries = []
    for key in sorted(consensus.get("stage3", {}).keys(), key=_tuple_sort_key):
        m, v = key
        vals = consensus["stage3"][key]
        stage3_entries.append({
            "m": _json_safe(m),
            "v": _json_safe(v),
            "x": float(vals["x"]),
            "r": float(vals["r"]),
        })

    return {
        "stage1": stage1_entries,
        "stage2": stage2_entries,
        "stage3": stage3_entries,
    }


def write_bidding_policy(consensus, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(consensus_to_json(consensus), handle, indent=2)


def run_progressive_hedging(time_str, n_total, n_per_bundle, num_bundles, seed=0, verbose=True,
                            alpha=100, epsilon=1e-2, max_iter=50, gap_pct=0.01,
                            adaptive_alpha=True, tau=2.0, mu=10.0,
                            bidding_output_dir: str | Path | None = None):
    """
    Entry point for the Progressive Hedging algorithm.

    Steps 1-11:  Initial solve, consensus, shadow costs.
    Steps 12-16: While g > epsilon, re-solve augmented bundles.

    Input:
        time_str:       timestamp string
        n_total:        total number of scenarios
        n_per_bundle:   number of scenarios per bundle tree
        num_bundles:    how many bundles to generate
        seed:           base seed for bundle generation
        verbose:        print progress info
        alpha:          PH penalty parameter (initial value)
        epsilon:        fallback convergence tolerance when objective mean is unavailable
        max_iter:       maximum number of PH iterations
        gap_pct:        convergence threshold as a fraction of |objective_unaugmented_mean|
        adaptive_alpha: if True, use residual-balancing to adapt alpha each iteration
        tau:            scaling factor for alpha adjustments (default 2.0)
        mu:             threshold ratio for triggering adjustment (default 10.0)
        bidding_output_dir: optional directory for writing a bidding policy file

    Returns:
        B:         list of scenario tree dicts
        results:   latest per-bundle result dicts
        consensus: latest consensus dict
        W_shadow:  latest shadow-cost dicts
    """

    input_data = read.load_parameters_from_parquet(time_str, n_total, seed)
    global_bounds = read.get_global_bounds_from_input_data(input_data)

    # Build scenario bundles
    B = tree.build_scenario_bundles(input_data, n_per_bundle, num_bundles, seed=seed)

    if verbose:
        adapt_str = f", adaptive (tau={tau}, mu={mu})" if adaptive_alpha else ", fixed"
        print(f"[PH] {num_bundles} bundles, {n_per_bundle} scenarios each, "
              f"alpha={alpha}{adapt_str}, eps={epsilon}")
        print_iteration_header()

    # ------------------------------------------------------------------
    # Pre-build all bundle models (reused across iterations)
    # ------------------------------------------------------------------
    if verbose:
        print("[PH] Pre-building bundle models...")
    models, base_objs = build_bundle_models(B, global_bounds)
    if verbose:
        print(f"[PH] Built {len(models)} models")

    # Fetch market products once, pass to all functions
    market_products = get_market_products()

    # ------------------------------------------------------------------
    # Steps 2-5: Initial solve (no penalty terms)
    # ------------------------------------------------------------------
    results = solve_bundles(B, global_bounds, market_products, models=models, base_objs=base_objs, verbose=False)

    # Steps 6-8: Compute consensus
    consensus = compute_consensus(results, verbose=False)

    # Steps 9-10: Initialise shadow costs
    W_shadow = initialize_shadow_costs(results, consensus, alpha=alpha, verbose=False)

    # Step 12: Compute initial convergence gap
    g = compute_convergence_gap(results, consensus, market_products)
    k = 0
    mean_obj = _objective_mean(results, key="objective_unaugmented")
    gap_threshold = _gap_threshold_from_objective(mean_obj, gap_pct, epsilon)
    effective_max_iter = min(int(max_iter), 100)

    if verbose:
        print_iteration_row(k, g, results, alpha=alpha)

    # ------------------------------------------------------------------
    # Steps 12-16: Iterative improvements
    # ------------------------------------------------------------------
    while g > gap_threshold and k < effective_max_iter:
        # Step 13: increment iteration counter
        k += 1

        # Steps 14-16: solve augmented sub-problems for every bundle
        results = solve_bundles_augmented(
            B, global_bounds, W_shadow, consensus, alpha,
            market_products, models=models, base_objs=base_objs, verbose=False
        )

        # Steps 18-20: Update consensus with the new individual decisions
        prev_consensus = consensus
        consensus = compute_consensus(results, verbose=False)

        # Steps 21-22: Update shadow costs
        W_shadow = update_shadow_costs(W_shadow, results, consensus, alpha)

        # Step 24: Recompute convergence gap
        g = compute_convergence_gap(results, consensus, market_products)
        mean_obj = _objective_mean(results, key="objective_unaugmented")
        gap_threshold = _gap_threshold_from_objective(mean_obj, gap_pct, epsilon)

        # Adaptive alpha: residual-balancing
        if adaptive_alpha:
            dual_res = compute_dual_residual(consensus, prev_consensus, alpha)
            alpha = adapt_alpha(alpha, g, dual_res, tau=tau, mu=mu)

        if verbose:
            print_iteration_row(k, g, results, alpha=alpha)

    if verbose:
        status = "CONVERGED" if g <= gap_threshold else f"MAX ITER ({effective_max_iter})"
        print(f"{'':->82}")
        print(f"  Terminated: {status}  (gap={g:.6f}, alpha={alpha:.4f})")
        print_final_consensus(consensus)

    if bidding_output_dir is not None:
        output_dir = Path(bidding_output_dir)
        safe_time = _sanitize_time_str(time_str)
        policy_path = output_dir / f"bidding_policy_pha_{safe_time}_{_run_stamp()}.json"
        write_bidding_policy(consensus, policy_path)
        if verbose:
            print(f"[PH] Bidding policy written to: {policy_path}")

    return B, results, consensus, W_shadow

def print_iteration_header():
    """Print the header row for the PH iteration table."""
    print(f"\n{'':=<82}")
    print(f"  {'Iter':>4s}  {'Gap':>12s}  {'Alpha':>10s}  {'Solved':>6s}  "
          f"{'Obj mean':>12s}  {'Obj std':>10s}  {'Obj range':>12s}")
    print(f"{'':->82}")


def print_iteration_row(k, gap, results, alpha=None):
    """Print one row summarising PH iteration k."""
    objs = [r["objective"] for r in results if r is not None]
    n_solved = len(objs)
    alpha_str = f"{alpha:>10.4f}" if alpha is not None else f"{'—':>10s}"
    if n_solved == 0:
        print(f"  {k:>4d}  {'inf':>12s}  {alpha_str}  {0:>6d}  {'—':>12s}  {'—':>10s}  {'—':>12s}")
        return
    mean_obj = sum(objs) / n_solved
    std_obj = (sum((o - mean_obj)**2 for o in objs) / n_solved) ** 0.5
    obj_range = max(objs) - min(objs)
    print(f"  {k:>4d}  {gap:>12.6f}  {alpha_str}  {n_solved:>6d}  "
          f"{mean_obj:>12.4f}  {std_obj:>10.4f}  {obj_range:>12.4f}")


def print_final_consensus(consensus):
    """Print a compact summary of the converged consensus decisions."""
    print(format_final_consensus(consensus), end="")