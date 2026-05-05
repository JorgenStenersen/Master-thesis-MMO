#!/usr/bin/env python3
"""
Run a benchmark: solve the extensive form once, then run Progressive Hedging
for up to 10 different (n_total, n_per_bundle, num_bundles) combinations.

Outputs a CSV with timing and objective info for the extensive run and each
PH configuration.

Usage examples:
    python experiments.run_ph_bench \
        --time-str "2025-04-04 08:00:00+00:00" \
        --time-str "2025-06-10 20:00:00+00:00" \
        --time-str "2025-08-20 13:00:00+00:00" \
        --time-str "2025-12-15 05:00:00+00:00" \
        --extensive-n 3 \
        --combo 3:1:3 --combo 3:1:6 \
        --out results/ph_bench.csv

Each --combo must be of the form n_total:n_per_bundle:num_bundles. Max 10 combos.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

from gurobipy import GRB

import src.read as read
import src.tree as tree
from src.model import build_model
from src.solvers import progressive_hedging
from scripts.ph_slurm_coordinator import run_distributed_ph


def parse_combo(s: str) -> Tuple[int, int, int]:
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("combo must be n_total:n_per_bundle:num_bundles")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise argparse.ArgumentTypeError("combo values must be integers")


def _sanitize_time_str(time_str: str) -> str:
    safe = []
    for ch in time_str:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _resolve_combo_index(explicit_idx: int | None) -> int | None:
    if explicit_idx is not None:
        return explicit_idx
    if "SGE_TASK_ID" in os.environ:
        sge_task_id = os.environ.get("SGE_TASK_ID", "")
        if sge_task_id and sge_task_id != "undefined":
            return int(sge_task_id) - 1
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
    return None


def _build_ph_work_dir(work_root: Path, time_str: str, combo: Tuple[int, int, int]) -> Path:
    n_total, n_per_bundle, num_bundles = combo
    safe_time = _sanitize_time_str(time_str)
    combo_tag = f"combo_{n_total}_{n_per_bundle}_{num_bundles}"
    return work_root / safe_time / combo_tag


def run_extensive(time_str: str, n: int, seed: int | None = None) -> dict:
    """Build and solve the extensive form once. Returns metrics dict."""
    start = time.perf_counter()

    input_data = read.load_parameters_from_parquet(time_str, n, seed)
    scenario_tree = tree.build_scenario_tree(input_data)
    global_bounds = read.get_global_bounds_from_input_data(input_data)

    mc = build_model(scenario_tree, global_bounds, mode="extensive")

    # Solve
    mc.model.optimize()

    end = time.perf_counter()
    elapsed = end - start

    status = getattr(mc.model, "Status", None)
    objective = None
    if status == GRB.OPTIMAL:
        # Objective is set for maximization
        objective = float(mc.model.ObjVal)

    return {
        "run_type": "extensive",
        "n_total": n,
        "n_per_bundle": "",
        "num_bundles": "",
        "runtime_seconds": elapsed,
        "objective_mean": objective,
        "objective_min": objective,
        "objective_max": objective,
        "status": status,
    }


def run_ph_combo_inprocess(time_str: str, n_total: int, n_per_bundle: int, num_bundles: int,
                           seed: int = 0, alpha: float = 100.0, epsilon: float = 1e-2,
                           max_iter: int = 50, gap_pct: float = 0.01, adaptive_alpha: bool = True,
                           tau: float = 2.0, mu: float = 10.0,
                           bidding_output_dir: Path | None = None) -> tuple[dict, str]:
    """Run progressive hedging for one combo, measure runtime and extract objective stats."""
    start = time.perf_counter()
    status = "ok"
    results = []
    try:
        _, results, _, _ = progressive_hedging.run_progressive_hedging(
            time_str=time_str,
            n_total=n_total,
            n_per_bundle=n_per_bundle,
            num_bundles=num_bundles,
            seed=seed,
            verbose=False,
            alpha=alpha,
            epsilon=epsilon,
            max_iter=max_iter,
            gap_pct=gap_pct,
            adaptive_alpha=adaptive_alpha,
            tau=tau,
            mu=mu,
            bidding_output_dir=bidding_output_dir,
        )
    except Exception as exc:  # pragma: no cover - runtime failures handled in reporting
        status = f"error: {type(exc).__name__}"

    end = time.perf_counter()
    elapsed = end - start

    objs = [r["objective"] for r in results if r is not None and "objective" in r]
    mean_obj = sum(objs) / len(objs) if objs else None
    min_obj = min(objs) if objs else None
    max_obj = max(objs) if objs else None

    return {
        "run_type": "ph",
        "n_total": n_total,
        "n_per_bundle": n_per_bundle,
        "num_bundles": num_bundles,
        "runtime_seconds": elapsed,
        "objective_mean": mean_obj,
        "objective_min": min_obj,
        "objective_max": max_obj,
        "status": status,
    }, "inprocess"


def run_ph_combo_coordinator(time_str: str, n_total: int, n_per_bundle: int, num_bundles: int,
                             seed: int, alpha: float, epsilon: float, max_iter: int,
                             gap_pct: float, adaptive_alpha: bool, tau: float, mu: float,
                             work_dir: Path, max_workers: int,
                             gurobi_threads_per_bundle: int) -> tuple[dict, str]:
    start = time.perf_counter()

    run_distributed_ph(
        time_str=time_str,
        n_total=n_total,
        n_per_bundle=n_per_bundle,
        num_bundles=num_bundles,
        seed=seed,
        alpha=alpha,
        epsilon=epsilon,
        max_iter=max_iter,
        gap_pct=gap_pct,
        adaptive_alpha=adaptive_alpha,
        tau=tau,
        mu=mu,
        work_dir=work_dir,
        max_workers=max_workers,
        gurobi_threads_per_bundle=gurobi_threads_per_bundle,
    )

    end = time.perf_counter()
    elapsed = end - start

    run_summary_path = work_dir / "run_summary.json"
    iter_summary = {}
    status = "unknown"
    if run_summary_path.exists():
        with run_summary_path.open("r", encoding="utf-8") as f:
            run_summary = json.load(f)
        status = run_summary.get("status", status)
        iteration_summaries = run_summary.get("iteration_summaries", [])
        if iteration_summaries:
            iter_summary = iteration_summaries[-1]

    return {
        "run_type": "ph",
        "n_total": n_total,
        "n_per_bundle": n_per_bundle,
        "num_bundles": num_bundles,
        "runtime_seconds": elapsed,
        "objective_mean": iter_summary.get("objective_mean"),
        "objective_min": iter_summary.get("objective_min"),
        "objective_max": iter_summary.get("objective_max"),
        "status": status,
    }, "coordinator"


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark extensive vs PH with multiple configs")
    parser.add_argument("--time-str", action="append", required=True,
                        help="Timestamp(s) to run the benchmark for. Specify one or more occurrences.")
    parser.add_argument("--extensive-n", type=int, default=None, help="n for the extensive form run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--combo", type=parse_combo, action="append",
                        help="Specify a PH combo as n_total:n_per_bundle:num_bundles (repeat up to 10 times)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--gap-pct", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--adaptive-alpha", type=int, choices=(0, 1), default=1)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--run-type", choices=("extensive", "ph", "both"), default="both")
    parser.add_argument("--ph-backend", choices=("inprocess", "coordinator"), default="inprocess")
    parser.add_argument("--results-root", type=str, default=os.environ.get("RESULTS_ROOT", "results"))
    parser.add_argument("--ph-work-root", type=str, default=os.environ.get("PH_WORK_ROOT", ""))
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--gurobi-threads-per-bundle", type=int, default=None)
    parser.add_argument("--combo-index", type=int, default=None,
                        help="Run only the selected combo index (0-based). If omitted, SGE/Slurm task ID is used.")

    args = parser.parse_args(argv)

    combos = args.combo or []
    if len(combos) > 10:
        raise SystemExit("A maximum of 10 combos is supported")

    time_strs = args.__dict__.get("time_str", [])
    if len(time_strs) == 0:
        raise SystemExit("Please specify at least one --time-str argument")

    if args.run_type in ("extensive", "both") and args.extensive_n is None:
        raise SystemExit("--extensive-n is required for extensive runs")

    if args.run_type in ("ph", "both") and not combos:
        raise SystemExit("At least one --combo is required for PH runs")

    combo_index = _resolve_combo_index(args.combo_index)
    if combo_index is not None:
        if combo_index < 0 or combo_index >= len(combos):
            raise SystemExit(f"combo index {combo_index} out of range (0..{len(combos) - 1})")
        combos = [combos[combo_index]]

    results_root = Path(args.results_root)
    out_path = Path(args.out) if args.out else (results_root / "ph_bench.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ph_work_root = Path(args.ph_work_root) if args.ph_work_root else (results_root / "ph_runs")
    max_workers = args.max_workers
    total_cores = max(1, os.cpu_count() or 1)
    if max_workers is None:
        max_workers = total_cores
    gurobi_threads_per_bundle = args.gurobi_threads_per_bundle
    if gurobi_threads_per_bundle is None:
        gurobi_threads_per_bundle = max(1, total_cores // max_workers)

    fieldnames = [
        "time_str",
        "run_type",
        "ph_backend",
        "n_total",
        "n_per_bundle",
        "num_bundles",
        "runtime_seconds",
        "objective_mean",
        "objective_min",
        "objective_max",
        "status",
        "ph_work_dir",
    ]

    rows = []

    # Run the workflow for each provided time_str
    for t_idx, t in enumerate(time_strs):
        print(f"[BENCH] === Time {t_idx+1}/{len(time_strs)}: {t} ===")

        if args.run_type in ("extensive", "both"):
            print(f"[BENCH] Running extensive form (n={args.extensive_n}) for time {t}")
            ext_metrics = run_extensive(t, args.extensive_n, seed=args.seed)
            ext_metrics["time_str"] = t
            ext_metrics["ph_backend"] = ""
            ext_metrics["ph_work_dir"] = ""
            rows.append(ext_metrics)
            print(
                f"[BENCH] Extensive finished (time {t}): runtime={ext_metrics['runtime_seconds']:.2f}s, "
                f"obj={ext_metrics['objective_mean']}"
            )

        if args.run_type in ("ph", "both"):
            for idx, combo in enumerate(combos):
                n_total, n_per_bundle, num_bundles = combo
                print(
                    f"[BENCH] Running PH config {idx+1}/{len(combos)} for time {t}: "
                    f"n_total={n_total}, n_per_bundle={n_per_bundle}, num_bundles={num_bundles}"
                )
                ph_work_dir = _build_ph_work_dir(ph_work_root, t, combo)
                if args.ph_backend == "coordinator":
                    ph_metrics, backend = run_ph_combo_coordinator(
                        time_str=t,
                        n_total=n_total,
                        n_per_bundle=n_per_bundle,
                        num_bundles=num_bundles,
                        seed=args.seed,
                        alpha=args.alpha,
                        epsilon=args.epsilon,
                        max_iter=args.max_iter,
                        gap_pct=args.gap_pct,
                        adaptive_alpha=bool(args.adaptive_alpha),
                        tau=args.tau,
                        mu=args.mu,
                        work_dir=ph_work_dir,
                        max_workers=max_workers,
                        gurobi_threads_per_bundle=gurobi_threads_per_bundle,
                    )
                else:
                    ph_metrics, backend = run_ph_combo_inprocess(
                        time_str=t,
                        n_total=n_total,
                        n_per_bundle=n_per_bundle,
                        num_bundles=num_bundles,
                        seed=args.seed,
                        alpha=args.alpha,
                        epsilon=args.epsilon,
                        max_iter=args.max_iter,
                        gap_pct=args.gap_pct,
                        adaptive_alpha=bool(args.adaptive_alpha),
                        tau=args.tau,
                        mu=args.mu,
                        bidding_output_dir=ph_work_root,
                    )

                ph_metrics["time_str"] = t
                ph_metrics["ph_backend"] = backend
                ph_metrics["ph_work_dir"] = str(ph_work_dir) if args.ph_backend == "coordinator" else ""
                rows.append(ph_metrics)
                print(
                    f"[BENCH] PH finished (time {t}): runtime={ph_metrics['runtime_seconds']:.2f}s, "
                    f"mean_obj={ph_metrics['objective_mean']}"
                )

    # Compute averages across the provided time points for each unique config
    from collections import defaultdict

    agg = defaultdict(list)
    for r in rows:
        if r.get("time_str") == "AVERAGE":
            continue
        key = (r["run_type"], r["ph_backend"], r["n_total"], r["n_per_bundle"], r["num_bundles"])
        agg[key].append(r)

    for key, group in agg.items():
        # Only average groups that have exactly one entry per time
        if len(group) == len(time_strs):
            runtimes = [g["runtime_seconds"] for g in group if g.get("runtime_seconds") is not None]
            mean_objs = [g["objective_mean"] for g in group if g.get("objective_mean") is not None]
            avg_row = {
                "time_str": "AVERAGE",
                "run_type": key[0],
                "ph_backend": key[1],
                "n_total": key[2],
                "n_per_bundle": key[3],
                "num_bundles": key[4],
                "runtime_seconds": (sum(runtimes) / len(runtimes)) if runtimes else "",
                "objective_mean": (sum(mean_objs) / len(mean_objs)) if mean_objs else "",
                "objective_min": "",
                "objective_max": "",
                "status": "avg",
                "ph_work_dir": "",
            }
            rows.append(avg_row)

    # Write CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # ensure numeric values are serializable
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})

    print(f"[BENCH] Results written to: {out_path}")


if __name__ == "__main__":
    main()