#!/usr/bin/env python3
"""
Run Progressive Hedging for multiple bundle combos at a single timestamp.

Default combos:
    4:1:576
    6:1:1944
    8:1:4608
    10:1:9000

Outputs a CSV with runtime and convergence metrics and removes each combo
work directory after its metrics are captured.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from scripts.ph_slurm_coordinator import run_distributed_ph


DEFAULT_TIME_STR = "2025-03-20 08:00:00+00:00"
DEFAULT_COMBOS = [
    (5, 1, 1125),
    (7, 1, 3087),
    (9, 1, 6561),
    (15, 1, 30375),
]


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


def _build_ph_work_dir(work_root: Path, time_str: str, combo: Tuple[int, int, int]) -> Path:
    n_total, n_per_bundle, num_bundles = combo
    safe_time = _sanitize_time_str(time_str)
    combo_tag = f"combo_{n_total}_{n_per_bundle}_{num_bundles}"
    return work_root / safe_time / combo_tag


def _read_run_summary(work_dir: Path) -> dict:
    summary_path = work_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_metrics(run_summary: dict) -> Tuple[float | None, float | None, float | None, str]:
    gap = run_summary.get("gap")
    status = run_summary.get("status", "unknown")
    iter_summaries = run_summary.get("iteration_summaries", [])
    avg_unaug = None
    if iter_summaries:
        avg_unaug = iter_summaries[-1].get("objective_unaugmented_mean")
    return gap, avg_unaug, run_summary.get("total_elapsed_seconds"), status


def _maybe_remove_empty_dir(path: Path) -> None:
    try:
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Progressive Hedging bundle combos for one timestamp"
    )
    parser.add_argument("--time-str", type=str, default=DEFAULT_TIME_STR)
    parser.add_argument("--combo", type=parse_combo, action="append")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--gap-pct", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--adaptive-alpha", type=int, choices=(0, 1), default=1)
    parser.add_argument("--tau", type=float, default=3.4)
    parser.add_argument("--mu", type=float, default=7.4)
    parser.add_argument("--results-root", type=str, default=os.path.join("results", "full_test"))
    parser.add_argument("--ph-work-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--gurobi-threads-per-bundle", type=int, default=None)

    args = parser.parse_args(argv)

    combos = args.combo or DEFAULT_COMBOS

    results_root = Path(args.results_root)
    ph_work_root = Path(args.ph_work_root) if args.ph_work_root else (results_root / "ph_runs")
    out_path = Path(args.out) if args.out else (results_root / "ph_combo_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_cores = max(1, os.cpu_count() or 1)
    max_workers = args.max_workers or total_cores
    gurobi_threads_per_bundle = args.gurobi_threads_per_bundle
    if gurobi_threads_per_bundle is None:
        gurobi_threads_per_bundle = max(1, total_cores // max_workers)

    run_timestamp_utc = datetime.now(timezone.utc).isoformat()

    fieldnames = [
        "time_str",
        "run_timestamp_utc",
        "Extensive n",
        "# Bundles",
        "Runtime PHA (s)",
        "Convergence gap",
        "Avg. unaugmented objective",
        "Status",
    ]

    rows = []

    print(f"[SWEEP] Time: {args.time_str}")
    print(f"[SWEEP] Combos: {', '.join(f'{a}:{b}:{c}' for a, b, c in combos)}")
    print(f"[SWEEP] Using max_workers={max_workers}, gurobi_threads_per_bundle={gurobi_threads_per_bundle}")

    for idx, combo in enumerate(combos, start=1):
        n_total, n_per_bundle, num_bundles = combo
        work_dir = _build_ph_work_dir(ph_work_root, args.time_str, combo)

        print(
            f"[SWEEP] Running combo {idx}/{len(combos)}: "
            f"n_total={n_total}, n_per_bundle={n_per_bundle}, num_bundles={num_bundles}"
        )

        start = time.perf_counter()
        status = "unknown"
        gap = None
        avg_unaug = None

        try:
            run_distributed_ph(
                time_str=args.time_str,
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
                work_dir=work_dir,
                max_workers=max_workers,
                gurobi_threads_per_bundle=gurobi_threads_per_bundle,
            )
        except Exception as exc:  # pragma: no cover - runtime failures handled in reporting
            status = f"error: {type(exc).__name__}"
        end = time.perf_counter()
        runtime_seconds = end - start

        run_summary = _read_run_summary(work_dir)
        if run_summary:
            gap, avg_unaug, _, status = _extract_metrics(run_summary)

        rows.append(
            {
                "time_str": args.time_str,
                "run_timestamp_utc": run_timestamp_utc,
                "Extensive n": n_total,
                "# Bundles": num_bundles,
                "Runtime PHA (s)": runtime_seconds,
                "Convergence gap": gap,
                "Avg. unaugmented objective": avg_unaug,
                "Status": status,
            }
        )

        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            _maybe_remove_empty_dir(work_dir.parent)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SWEEP] Results written to: {out_path}")


if __name__ == "__main__":
    main()
