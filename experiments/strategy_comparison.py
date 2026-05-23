from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import time
from typing import Iterable

import src.solvers.extensive_form as extensive_form
import src.solvers.progressive_hedging as progressive_hedging


def _sanitize_for_filename(value: str) -> str:
	return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in value)


def hourly_timestamps_2025() -> list[str]:
	"""Return all hourly UTC timestamps in 2025 as strings compatible with read.load_parameters_from_parquet.

	Format used: 'YYYY-MM-DD HH:MM:SS+00:00' (explicit UTC offset)
	"""
	start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
	hours = []
	for h in range(365 * 24):
		ts = start + timedelta(hours=h)
		# Use space-separated format with explicit +00:00 to match other code
		hours.append(ts.strftime("%Y-%m-%d %H:%M:%S+00:00"))
	return hours


def sample_timepoints(k: int, seed: int | None = None) -> list[str]:
	pool = hourly_timestamps_2025()
	rnd = random.Random(seed)
	if k >= len(pool):
		return list(pool)
	return rnd.sample(pool, k)


def _find_latest_extensive_result(time_str: str, n: int) -> Path | None:
	"""Search results/extensive for the latest file matching the time_str and n convention.
	Returns Path or None if not found.
	"""
	safe = _sanitize_for_filename(time_str)
	results_root = Path("results") / "extensive"
	if not results_root.exists():
		return None
	pattern = f"extensive_{safe}_n{n}_*.json"
	candidates = list(results_root.glob(pattern))
	if not candidates:
		# Try a more permissive search (time_str may have slightly different formatting)
		candidates = [p for p in results_root.iterdir() if f"_n{n}_" in p.name and safe in p.name]
	if not candidates:
		return None
	candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	return candidates[0]


def generate_strategies(
	num_samples: int,
	extensive_n: int,
	ph_n_total: int,
	ph_n_per_bundle: int,
	ph_num_bundles: int,
	seed: int | None = None,
	output_root: str | Path = "results/strategies",
):
	"""Sample hourly timepoints in 2025 and generate strategies.

	For each sampled timepoint this function:
	- runs extensive form (calls src.solvers.extensive_form.run_model)
	  and copies the produced JSON into output_root/{safe_time}/extensive/
	- runs progressive hedging (calls src.solvers.progressive_hedging.run_progressive_hedging)
	  and writes PH consensus and results into output_root/{safe_time}/ph/

	Files are organised so later analysis can easily load all strategies per method
	and compare them pairwise for each timepoint.
	"""
	out_root = Path(output_root)
	out_root.mkdir(parents=True, exist_ok=True)

	times = sample_timepoints(num_samples, seed=seed)

	summary = []

	for idx, t in enumerate(times, start=1):
		print(f"[{idx}/{len(times)}] Processing time: {t}")
		safe_time = _sanitize_for_filename(t)
		time_dir = out_root / safe_time
		extensive_dir = time_dir / "extensive"
		ph_dir = time_dir / "ph"
		extensive_dir.mkdir(parents=True, exist_ok=True)
		ph_dir.mkdir(parents=True, exist_ok=True)

		meta = {
			"time_str": t,
			"timestamp_utc": datetime.now(timezone.utc).isoformat(),
			"seed": seed,
			"extensive_n": extensive_n,
			"ph_n_total": ph_n_total,
			"ph_n_per_bundle": ph_n_per_bundle,
			"ph_num_bundles": ph_num_bundles,
		}

		# --- Extensive form ---
		try:
			print("  Running extensive-form solver...")
			extensive_form.run_model(t, extensive_n, seed=None, verbose=False)
		except Exception as exc:
			print(f"  [ERROR] Extensive run failed for {t}: {exc}")
			meta["extensive_error"] = f"{type(exc).__name__}: {exc}"
			# continue to PH even if extensive fails
		else:
			# Find generated extensive result and copy it into our folder
			src_path = _find_latest_extensive_result(t, extensive_n)
			if src_path is not None:
				dst = extensive_dir / src_path.name
				try:
					shutil.copy2(src_path, dst)
					meta["extensive_result"] = str(dst)
				except Exception as exc:
					print(f"  [WARN] Failed to copy extensive result: {exc}")
			else:
				print("  [WARN] Could not locate extensive result file after run")

		# --- Progressive Hedging ---
		try:
			print("  Running progressive hedging (this may take a while)...")
			# Let PH write bidding/consensus file into our ph_dir by passing bidding_output_dir
			B, results, consensus, W_shadow = progressive_hedging.run_progressive_hedging(
				time_str=t,
				n_total=ph_n_total,
				n_per_bundle=ph_n_per_bundle,
				num_bundles=ph_num_bundles,
				seed=seed or 0,
				verbose=False,
				bidding_output_dir=ph_dir,
			)
			# Save raw results and consensus for later analysis
			try:
				with (ph_dir / "ph_results_raw.json").open("w", encoding="utf-8") as f:
					json.dump({"B": B, "results": results}, f, default=str, indent=2)
				with (ph_dir / "ph_consensus_raw.json").open("w", encoding="utf-8") as f:
					json.dump({"consensus": consensus}, f, default=str, indent=2)
				meta["ph_written"] = True
			except Exception as exc:
				print(f"  [WARN] Failed to write PH artifacts: {exc}")
		except Exception as exc:
			print(f"  [ERROR] Progressive Hedging failed for {t}: {exc}")
			meta["ph_error"] = f"{type(exc).__name__}: {exc}"

		# Write metadata for this timepoint
		with (time_dir / "metadata.json").open("w", encoding="utf-8") as mf:
			json.dump(meta, mf, indent=2)

		summary.append(meta)

		# be kind to the filesystem / avoid bursts
		time.sleep(0.1)

	# Write overall summary
	with (out_root / "summary.json").open("w", encoding="utf-8") as sf:
		json.dump({"requested_samples": num_samples, "generated": len(summary), "items": summary}, sf, indent=2)

	print(f"Done. Strategies written under: {out_root}")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Generate extensive-form and PH strategies for random hours in 2025")
	parser.add_argument("--num", type=int, default=200, help="Number of random timepoints to sample")
	parser.add_argument("--extensive-n", type=int, default=4, help="n parameter for extensive-form run")
	parser.add_argument("--ph-n-total", type=int, default=4, help="PH total scenarios (n_total)")
	parser.add_argument("--ph-n-per-bundle", type=int, default=1, help="PH scenarios per bundle (n_per_bundle)")
	parser.add_argument("--ph-num-bundles", type=int, default=352, help="PH number of bundles")
	parser.add_argument("--seed", type=int, default=30, help="Random seed for sampling and PH")
	parser.add_argument("--out", type=str, default="results/strategies", help="Output root directory")

	args = parser.parse_args()

	generate_strategies(
		num_samples=args.num,
		extensive_n=args.extensive_n,
		ph_n_total=args.ph_n_total,
		ph_n_per_bundle=args.ph_n_per_bundle,
		ph_num_bundles=args.ph_num_bundles,
		seed=args.seed,
		output_root=args.out,
	)

