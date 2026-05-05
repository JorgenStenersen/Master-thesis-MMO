import argparse
import csv
import json
import pickle
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import src.read as read
import src.tree as tree
from src.model import get_market_products
from src.solvers.progressive_hedging import (
    adapt_alpha,
    compute_consensus,
    compute_convergence_gap,
    compute_dual_residual,
    initialize_shadow_costs,
    print_final_consensus,
    write_bidding_policy,
    print_iteration_header,
    print_iteration_row,
    update_shadow_costs,
)
from scripts.ph_bundle_worker import run_bundle_job


def _save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_time_str(time_str: str) -> str:
    safe = []
    for ch in time_str:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _gap_threshold_from_objective(objective_mean: float | None, gap_pct: float, epsilon: float) -> float:
    if objective_mean is None:
        return epsilon
    if not math.isfinite(objective_mean):
        return epsilon
    scaled = gap_pct * abs(objective_mean)
    if scaled <= 0.0:
        return epsilon
    return scaled


def _run_bundle_batch(mode: str, iteration: int, num_bundles: int, static_file: Path,
                      out_dir: Path, max_workers: int, gurobi_threads: int,
                      consensus: dict | None = None,
                      w_shadow_list: list | None = None,
                      alpha: float | None = None) -> tuple[list, dict]:
    start_ts = time.perf_counter()
    start_utc = _utc_now_iso()

    iter_dir = out_dir / f"iter_{iteration:03d}"
    (iter_dir / "logs").mkdir(parents=True, exist_ok=True)

    workers = max(1, min(int(max_workers), num_bundles))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for b_idx in range(num_bundles):
            w_shadow = None
            if w_shadow_list is not None:
                w_shadow = w_shadow_list[b_idx]
            futures.append(
                executor.submit(
                    run_bundle_job,
                    mode,
                    b_idx,
                    iteration,
                    str(static_file),
                    str(out_dir),
                    consensus,
                    w_shadow,
                    alpha,
                    int(gurobi_threads),
                )
            )

        results = [None] * num_bundles
        metrics = [None] * num_bundles
        for fut in as_completed(futures):
            result, metric = fut.result()
            idx = metric.get("bundle_index", None)
            if idx is None:
                continue
            results[idx] = result
            metrics[idx] = metric
    _write_iteration_timing_summary(iter_dir, iteration, num_bundles, metrics)

    end_ts = time.perf_counter()
    end_utc = _utc_now_iso()
    elapsed = end_ts - start_ts

    objectives = [r["objective"] for r in results if r is not None and "objective" in r]
    unaug_objectives = [
        r.get("objective_unaugmented")
        for r in results
        if r is not None and r.get("objective_unaugmented") is not None
    ]
    obj_min = min(objectives) if objectives else None
    obj_max = max(objectives) if objectives else None
    obj_mean = None
    obj_std = None
    if objectives:
        obj_mean = sum(objectives) / len(objectives)
        obj_std = (sum((o - obj_mean) ** 2 for o in objectives) / len(objectives)) ** 0.5

    obj_unaug_min = min(unaug_objectives) if unaug_objectives else None
    obj_unaug_max = max(unaug_objectives) if unaug_objectives else None
    obj_unaug_mean = None
    obj_unaug_std = None
    if unaug_objectives:
        obj_unaug_mean = sum(unaug_objectives) / len(unaug_objectives)
        obj_unaug_std = (
            sum((o - obj_unaug_mean) ** 2 for o in unaug_objectives) / len(unaug_objectives)
        ) ** 0.5

    iter_summary = {
        "iteration": iteration,
        "mode": mode,
        "bundles_total": num_bundles,
        "bundles_solved": sum(1 for r in results if r is not None),
        "bundles_missing": sum(1 for r in results if r is None),
        "elapsed_seconds": elapsed,
        "gurobi_threads_per_bundle": gurobi_threads,
        "objective_min": obj_min,
        "objective_max": obj_max,
        "objective_mean": obj_mean,
        "objective_std": obj_std,
        "objective_unaugmented_min": obj_unaug_min,
        "objective_unaugmented_max": obj_unaug_max,
        "objective_unaugmented_mean": obj_unaug_mean,
        "objective_unaugmented_std": obj_unaug_std,
        "start_utc": start_utc,
        "end_utc": end_utc,
    }

    return results, iter_summary


def _write_iteration_timing_summary(iter_dir: Path, iteration: int, num_bundles: int,
                                    metrics: list | None) -> None:
    logs_dir = iter_dir / "logs"
    summary_path = logs_dir / "timing_summary.csv"

    fieldnames = [
        "iteration",
        "bundle_index",
        "mode",
        "status",
        "elapsed_seconds",
        "gurobi_threads",
        "hostname",
        "pid",
        "sge_job_id",
        "sge_task_id",
        "start_utc",
        "end_utc",
        "error",
    ]

    rows = []
    for b_idx in range(num_bundles):
        row = None
        if metrics is not None and b_idx < len(metrics):
            row = metrics[b_idx]
        if not row:
            rows.append(
                {
                    "iteration": iteration,
                    "bundle_index": b_idx,
                    "mode": "",
                    "status": "missing",
                    "elapsed_seconds": "",
                    "gurobi_threads": "",
                    "hostname": "",
                    "pid": "",
                    "sge_job_id": "",
                    "sge_task_id": "",
                    "start_utc": "",
                    "end_utc": "",
                    "error": "timing file missing",
                }
            )
            continue

        rows.append({key: row.get(key, "") for key in fieldnames})

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_distributed_ph(time_str: str, n_total: int, n_per_bundle: int, num_bundles: int,
                       seed: int, alpha: float, epsilon: float, max_iter: int, gap_pct: float,
                       adaptive_alpha: bool, tau: float, mu: float, work_dir: Path,
                       max_workers: int, gurobi_threads_per_bundle: int) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)

    run_start_ts = time.perf_counter()
    run_start_utc = _utc_now_iso()
    iteration_summaries = []

    print("[PH-SGE] Loading data and building bundles...")
    input_data = read.load_parameters_from_parquet(time_str, n_total, seed)
    global_bounds = read.get_global_bounds_from_input_data(input_data)
    bundles = tree.build_scenario_bundles(input_data, n_per_bundle, num_bundles, seed=seed)
    market_products = get_market_products()

    static_file = work_dir / "static.pkl"
    _save_pickle(
        static_file,
        {
            "bundles": bundles,
            "global_bounds": global_bounds,
            "market_products": market_products,
        },
    )

    print_iteration_header()

    print(
        f"[PH-SGE] Local parallel execution with max_workers={max_workers}, "
        f"gurobi_threads_per_bundle={gurobi_threads_per_bundle}"
    )

    # Initial solves (k=0)
    results, iter_summary = _run_bundle_batch(
        mode="initial",
        iteration=0,
        num_bundles=num_bundles,
        static_file=static_file,
        out_dir=work_dir,
        max_workers=max_workers,
        gurobi_threads=gurobi_threads_per_bundle,
        consensus=None,
        w_shadow_list=None,
        alpha=None,
    )
    consensus = compute_consensus(results, verbose=False)
    w_shadow = initialize_shadow_costs(results, consensus, alpha=alpha, verbose=False)
    gap = compute_convergence_gap(results, consensus, market_products)
    k = 0
    gap_threshold = _gap_threshold_from_objective(
        iter_summary.get("objective_unaugmented_mean"),
        gap_pct,
        epsilon,
    )
    effective_max_iter = min(int(max_iter), 100)

    iter_summary.update(
        {
            "gap": gap,
            "alpha": alpha,
            "adaptive_alpha": adaptive_alpha,
            "tau": tau,
            "mu": mu,
            "gap_threshold": gap_threshold,
            "gap_pct": gap_pct,
        }
    )
    iter_dir = work_dir / f"iter_{k:03d}"
    _save_json(iter_dir / "summary.json", iter_summary)
    _save_json(iter_dir / "results_summary.json", {
        "iteration": k,
        "objective_min": iter_summary.get("objective_min"),
        "objective_max": iter_summary.get("objective_max"),
        "objective_mean": iter_summary.get("objective_mean"),
        "objective_std": iter_summary.get("objective_std"),
        "objective_unaugmented_min": iter_summary.get("objective_unaugmented_min"),
        "objective_unaugmented_max": iter_summary.get("objective_unaugmented_max"),
        "objective_unaugmented_mean": iter_summary.get("objective_unaugmented_mean"),
        "objective_unaugmented_std": iter_summary.get("objective_unaugmented_std"),
        "bundles_solved": iter_summary.get("bundles_solved"),
        "bundles_missing": iter_summary.get("bundles_missing"),
    })
    iteration_summaries.append(iter_summary)

    print_iteration_row(k, gap, results, alpha=alpha)

    while gap > gap_threshold and k < effective_max_iter:
        k += 1

        results, iter_summary = _run_bundle_batch(
            mode="augmented",
            iteration=k,
            num_bundles=num_bundles,
            static_file=static_file,
            out_dir=work_dir,
            max_workers=max_workers,
            gurobi_threads=gurobi_threads_per_bundle,
            consensus=consensus,
            w_shadow_list=w_shadow,
            alpha=alpha,
        )

        prev_consensus = consensus
        consensus = compute_consensus(results, verbose=False)
        w_shadow = update_shadow_costs(w_shadow, results, consensus, alpha)
        gap = compute_convergence_gap(results, consensus, market_products)
        gap_threshold = _gap_threshold_from_objective(
            iter_summary.get("objective_unaugmented_mean"),
            gap_pct,
            epsilon,
        )

        if adaptive_alpha:
            dual_res = compute_dual_residual(consensus, prev_consensus, alpha)
            alpha = adapt_alpha(alpha, gap, dual_res, tau=tau, mu=mu)

        iter_summary.update(
            {
                "gap": gap,
                "alpha": alpha,
                "adaptive_alpha": adaptive_alpha,
                "tau": tau,
                "mu": mu,
                "gap_threshold": gap_threshold,
                "gap_pct": gap_pct,
            }
        )
        iter_dir = work_dir / f"iter_{k:03d}"
        _save_json(iter_dir / "summary.json", iter_summary)
        _save_json(iter_dir / "results_summary.json", {
            "iteration": k,
            "objective_min": iter_summary.get("objective_min"),
            "objective_max": iter_summary.get("objective_max"),
            "objective_mean": iter_summary.get("objective_mean"),
            "objective_std": iter_summary.get("objective_std"),
            "objective_unaugmented_min": iter_summary.get("objective_unaugmented_min"),
            "objective_unaugmented_max": iter_summary.get("objective_unaugmented_max"),
            "objective_unaugmented_mean": iter_summary.get("objective_unaugmented_mean"),
            "objective_unaugmented_std": iter_summary.get("objective_unaugmented_std"),
            "bundles_solved": iter_summary.get("bundles_solved"),
            "bundles_missing": iter_summary.get("bundles_missing"),
        })
        iteration_summaries.append(iter_summary)

        print_iteration_row(k, gap, results, alpha=alpha)

    status = "CONVERGED" if gap <= gap_threshold else f"MAX ITER ({effective_max_iter})"
    print(f"{'':->82}")
    print(f"  Terminated: {status}  (gap={gap:.6f}, alpha={alpha:.4f})")
    print_final_consensus(consensus)

    policy_dir = work_dir.parent if work_dir.name.startswith("combo_") else work_dir
    policy_name = f"bidding_policy_pha_{_sanitize_time_str(time_str)}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    policy_path = policy_dir / policy_name
    write_bidding_policy(consensus, policy_path)
    print(f"[PH-SGE] Bidding policy written to: {policy_path}")

    _save_pickle(
        work_dir / "final_state.pkl",
        {
            "status": status,
            "iterations": k,
            "gap": gap,
            "alpha": alpha,
            "consensus": consensus,
            "results": results,
            "W_shadow": w_shadow,
        },
    )
    run_end_ts = time.perf_counter()
    run_end_utc = _utc_now_iso()
    run_summary = {
        "status": status,
        "iterations": k,
        "gap": gap,
        "alpha": alpha,
        "total_elapsed_seconds": run_end_ts - run_start_ts,
        "start_utc": run_start_utc,
        "end_utc": run_end_utc,
        "time_str": time_str,
        "n_total": n_total,
        "n_per_bundle": n_per_bundle,
        "num_bundles": num_bundles,
        "seed": seed,
        "adaptive_alpha": adaptive_alpha,
        "tau": tau,
        "mu": mu,
        "gap_threshold": gap_threshold,
        "gap_pct": gap_pct,
        "epsilon": epsilon,
        "iteration_summaries": iteration_summaries,
    }
    _save_json(work_dir / "run_summary.json", run_summary)
    print(f"[PH-SGE] Final state written to: {work_dir / 'final_state.pkl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-parallel PH coordinator for SGE jobs.")
    parser.add_argument("--time-str", type=str, required=True)
    parser.add_argument("--n-total", type=int, required=True)
    parser.add_argument("--n-per-bundle", type=int, required=True)
    parser.add_argument("--num-bundles", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--gap-pct", type=float, default=0.01)
    parser.add_argument("--adaptive-alpha", type=int, default=1)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--gurobi-threads-per-bundle", type=int, required=True)
    args = parser.parse_args()

    run_distributed_ph(
        time_str=args.time_str,
        n_total=args.n_total,
        n_per_bundle=args.n_per_bundle,
        num_bundles=args.num_bundles,
        seed=args.seed,
        alpha=args.alpha,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        gap_pct=args.gap_pct,
        adaptive_alpha=bool(args.adaptive_alpha),
        tau=args.tau,
        mu=args.mu,
        work_dir=Path(args.work_dir),
        max_workers=args.max_workers,
        gurobi_threads_per_bundle=args.gurobi_threads_per_bundle,
    )


if __name__ == "__main__":
    main()
