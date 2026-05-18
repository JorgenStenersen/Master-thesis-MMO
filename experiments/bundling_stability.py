#!/usr/bin/env python3
"""
Stability test for bundling randomness in Progressive Hedging.

Run PH several times for a set of timestamps and collect statistics over
the non-augmented objective values ("objective_unaugmented") returned
for each bundle run. This shows how much solutions vary due to random
bundling.

Usage example:
    python experiments/bundling_stability.py \
        --time-str "2025-03-20 08:00:00+00:00" \
        --time-str "2025-06-12 20:00:00+00:00" \
        --time-str "2025-08-22 13:00:00+00:00" \
        --time-str "2025-12-17 05:00:00+00:00" \
        --runs 10 \
        --n-total 50 --n-per-bundle 5 --num-bundles 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.solvers.progressive_hedging import run_progressive_hedging


def stats_from_list(values: List[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "variance": None,
            "stdev": None,
            "min": None,
            "max": None,
            "median": None,
        }
    n = len(values)
    mean = statistics.mean(values)
    var = statistics.pvariance(values) if n > 1 else 0.0
    stdev = statistics.pstdev(values) if n > 1 else 0.0
    return {
        "count": n,
        "mean": mean,
        "variance": var,
        "stdev": stdev,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def _sanitize_time_str(time_str: str) -> str:
    safe = []
    for ch in time_str:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def run_stability(time_str: str, runs: int, n_total: int, n_per_bundle: int, num_bundles: int,
                  ph_kwargs: dict | None = None) -> dict:
    ph_kwargs = ph_kwargs or {}
    objs = []
    runtimes = []

    for i in range(runs):
        # Use disjoint seed ranges per run; bundles within a run use seed + b.
        seed = 1 + i * num_bundles
        start = time.perf_counter()
        _, results, _, _ = run_progressive_hedging(
            time_str=time_str,
            n_total=n_total,
            n_per_bundle=n_per_bundle,
            num_bundles=num_bundles,
            seed=seed,
            verbose=False,
            **ph_kwargs,
        )
        end = time.perf_counter()
        runtimes.append(end - start)

        # Collect all unaugmented objectives from bundles for this run
        run_vals = [r.get("objective_unaugmented") for r in results if r is not None and r.get("objective_unaugmented") is not None]
        if not run_vals:
            # nothing solved; skip
            continue
        # For stability across bundling we compare mean (or sum) per-run.
        # Use mean of unaugmented objectives across bundles for this run.
        run_mean = sum(run_vals) / len(run_vals)
        objs.append(run_mean)

    return {
        "time_str": time_str,
        "runs_requested": runs,
        "runs_completed": len(objs),
        "objective_stats": stats_from_list(objs),
        "runtimes_seconds": {
            "count": len(runtimes),
            "mean": statistics.mean(runtimes) if runtimes else None,
            "median": statistics.median(runtimes) if runtimes else None,
        },
    }


def main(argv: List[str] | None = None) -> None:
    # Set run parameters here in Python instead of via CLI
    # Edit the following values as needed.
    time_strs = [
        "2025-03-20 08:00:00+00:00",
        "2025-06-12 20:00:00+00:00",
        "2025-08-22 13:00:00+00:00",
        "2025-12-17 05:00:00+00:00",
    ]

    runs = 10
    n_total = 4
    n_per_bundle = 1
    num_bundles = 576

    # PH parameters (tweak here)
    ph_kwargs = {
        "alpha": 100.0,
        "epsilon": 1e-2,
        "max_iter": 50,
        "gap_pct": 0.01,
        "adaptive_alpha": True,
        "tau": 3.4,
        "mu": 7.4,
    }

    # Output directory (one JSON per timestamp)
    out_dir = Path("results") / "bundling_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    results = []
    for t in time_strs:
        print(f"Running stability for {t} ({runs} runs)")
        res = run_stability(t, runs, n_total, n_per_bundle, num_bundles, ph_kwargs)
        results.append(res)
        print(json.dumps(res, indent=2))
        safe_time = _sanitize_time_str(t)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{safe_time}_{stamp}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
