from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TIME_CANDIDATES = ["time", "prediction_for", "created_at", "timestamp", "datetime"]
DEFAULT_OUTPUT_ROOT = "reduced_data_26"
DEFAULT_TARGET_SCENARIOS = 20
DEFAULT_MARKETS = [
	"dayahead",
	"mfrr_cm_up",
	"mfrr_cm_down",
	"mfrr_eam_up",
	"mfrr_eam_down",
	"production",
]


@dataclass
class ReductionResult:
	date: str
	hour: int
	time_column: str
	input_rows: int
	input_scenarios: int
	reduced_scenarios: int
	kantorovich_distance: float
	kept_columns_original: list[str]
	output_columns: list[str]
	probabilities: list[float]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Reduce forecast scenarios with a backward elimination algorithm."
	)
	parser.add_argument(
		"--input-root",
		default="updated_data_26",
		help="Root directory containing parquet files to reduce.",
	)
	parser.add_argument(
		"--output-root",
		default=DEFAULT_OUTPUT_ROOT,
		help="Directory where reduced parquet files will be written.",
	)
	parser.add_argument(
		"--target-scenarios",
		type=int,
		default=DEFAULT_TARGET_SCENARIOS,
		help="Number of scenarios to keep per hourly slice.",
	)
	parser.add_argument(
		"--file",
		default=None,
		help="Optional single parquet file to process instead of scanning the input root.",
	)
	parser.add_argument(
		"--timestamp",
		default=None,
		help="Optional UTC timestamp to reduce (ISO 8601, e.g. 2025-01-01T00:00:00Z).",
	)
	parser.add_argument(
		"--date",
		default=None,
		help="Optional date to reduce (YYYY-MM-DD).",
	)
	parser.add_argument(
		"--hour",
		type=int,
		default=None,
		help="Optional hour (0-23) to reduce.",
	)
	parser.add_argument(
		"--markets",
		nargs="*",
		default=DEFAULT_MARKETS,
		help="Market subdirectories under input root to include. Use 'all' to include every market.",
	)
	return parser.parse_args()


def parse_timestamp(value: str) -> tuple[str, int]:
	timestamp = pd.to_datetime(value, utc=True)
	if timestamp is pd.NaT:
		raise ValueError("timestamp could not be parsed")
	return timestamp.strftime("%Y-%m-%d"), int(timestamp.hour)


def find_time_column(columns: Iterable[str]) -> str | None:
	for candidate in TIME_CANDIDATES:
		if candidate in columns:
			return candidate
	return None


def get_scenario_columns(frame: pd.DataFrame) -> list[str]:
	scenario_columns: list[str] = []
	for column in frame.columns:
		try:
			int(str(column))
		except Exception:
			continue
		scenario_columns.append(str(column))
	return sorted(scenario_columns, key=lambda value: int(value))


def load_parquet_frame(path: Path) -> pd.DataFrame:
	frame = pd.read_parquet(path)
	if "__index_level_0__" in frame.columns:
		frame = frame.drop(columns=["__index_level_0__"])
	return frame


def normalize_output_filename(input_path: Path) -> str:
	name = input_path.name
	if name.endswith(".parquet"):
		name = name.replace("_PT1H", "")
	return name


def pairwise_cost_matrix(scenarios: np.ndarray) -> np.ndarray:
	"""Return the pairwise Manhattan distance matrix for scenario columns.

	The input must be shaped as (n_features, n_scenarios).
	"""

	if scenarios.ndim != 2:
		raise ValueError("scenarios must be a 2D array")

	n_scenarios = scenarios.shape[1]
	cost_matrix = np.zeros((n_scenarios, n_scenarios), dtype=float)

	for i in range(n_scenarios):
		for j in range(i + 1, n_scenarios):
			diff = scenarios[:, i] - scenarios[:, j]
			distance = float(np.sum(np.abs(diff)))
			cost_matrix[i, j] = distance
			cost_matrix[j, i] = distance

	return cost_matrix


def nearest_two(cost_matrix: np.ndarray, active_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Return nearest and second-nearest active keeper for every scenario."""

	active_indices = np.asarray(active_indices, dtype=int)
	active_costs = cost_matrix[:, active_indices]
	n_rows, n_active = active_costs.shape

	if n_active == 0:
		raise ValueError("active_indices must contain at least one scenario")

	if n_active == 1:
		nearest_positions = np.zeros(n_rows, dtype=int)
		nearest_distances = active_costs[:, 0].astype(float, copy=False)
		second_indices = np.full(n_rows, -1, dtype=int)
		second_distances = np.full(n_rows, np.inf, dtype=float)
		nearest_indices = active_indices[nearest_positions]
		return nearest_indices, nearest_distances, second_indices, second_distances

	top_two = np.argpartition(active_costs, kth=1, axis=1)[:, :2]
	top_two_values = np.take_along_axis(active_costs, top_two, axis=1)
	swap_mask = top_two_values[:, 0] > top_two_values[:, 1]

	nearest_positions = np.where(swap_mask, top_two[:, 1], top_two[:, 0])
	second_positions = np.where(swap_mask, top_two[:, 0], top_two[:, 1])
	nearest_distances = np.where(swap_mask, top_two_values[:, 1], top_two_values[:, 0]).astype(float)
	second_distances = np.where(swap_mask, top_two_values[:, 0], top_two_values[:, 1]).astype(float)

	nearest_indices = active_indices[nearest_positions]
	second_indices = active_indices[second_positions]
	return nearest_indices, nearest_distances, second_indices, second_distances


def candidate_loss(
	probabilities: np.ndarray,
	nearest_indices: np.ndarray,
	nearest_distances: np.ndarray,
	second_distances: np.ndarray,
	candidate_index: int,
) -> float:
	affected = nearest_indices == candidate_index
	if not np.any(affected):
		return 0.0
	delta = second_distances[affected] - nearest_distances[affected]
	return float(np.sum(probabilities[affected] * delta))


def backward_reduce(
	scenarios: np.ndarray,
	target_scenarios: int,
	probabilities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
	"""Reduce scenario columns with a backward elimination algorithm.

	Parameters
	----------
	scenarios:
		Matrix shaped as (n_features, n_scenarios).
	target_scenarios:
		Number of scenarios to keep.
	probabilities:
		Optional initial scenario probabilities.
	"""

	if scenarios.ndim != 2:
		raise ValueError("scenarios must be a 2D array")

	n_scenarios = scenarios.shape[1]
	if n_scenarios == 0:
		raise ValueError("scenarios must contain at least one scenario")

	target_scenarios = int(target_scenarios)
	if target_scenarios < 1:
		raise ValueError("target_scenarios must be at least 1")
	target_scenarios = min(target_scenarios, n_scenarios)

	if probabilities is None:
		probabilities = np.full(n_scenarios, 1.0 / n_scenarios, dtype=float)
	else:
		probabilities = np.asarray(probabilities, dtype=float)
		if probabilities.shape != (n_scenarios,):
			raise ValueError("probabilities must match the number of scenarios")
		total_probability = float(probabilities.sum())
		if not np.isclose(total_probability, 1.0):
			probabilities = probabilities / total_probability

	cost_matrix = pairwise_cost_matrix(scenarios)
	active_indices = np.arange(n_scenarios, dtype=int)

	while active_indices.size > target_scenarios:
		nearest_indices, nearest_distances, _, second_distances = nearest_two(cost_matrix, active_indices)

		losses = np.empty(active_indices.size, dtype=float)
		for position, candidate_index in enumerate(active_indices):
			losses[position] = candidate_loss(
				probabilities,
				nearest_indices,
				nearest_distances,
				second_distances,
				candidate_index,
			)

		remove_position = int(np.argmin(losses))
		active_indices = np.delete(active_indices, remove_position)

	nearest_indices, nearest_distances, _, _ = nearest_two(cost_matrix, active_indices)

	active_positions = np.full(n_scenarios, -1, dtype=int)
	active_positions[active_indices] = np.arange(active_indices.size, dtype=int)
	assignment_positions = active_positions[nearest_indices]

	reduced_probabilities = np.bincount(
		assignment_positions,
		weights=probabilities,
		minlength=active_indices.size,
	).astype(float)
	kantorovich_distance = float(np.sum(probabilities * nearest_distances))

	return active_indices, reduced_probabilities, kantorovich_distance, nearest_indices, nearest_distances


def reduce_hour_frame(
	frame: pd.DataFrame,
	scenario_columns: Sequence[str],
	target_scenarios: int,
	date: str,
	hour: int,
	time_column: str,
) -> tuple[pd.DataFrame, ReductionResult, list[str]]:
	metadata_columns = [column for column in frame.columns if column not in scenario_columns]
	scenario_matrix = frame.loc[:, scenario_columns].to_numpy(dtype=float)

	kept_indices, reduced_probabilities, kantorovich_distance, _, _ = backward_reduce(
		scenario_matrix,
		target_scenarios=target_scenarios,
	)

	kept_columns = [scenario_columns[index] for index in kept_indices]
	reduced_frame = frame.loc[:, metadata_columns + kept_columns].copy()

	output_columns = [str(index) for index in range(len(kept_columns))]
	rename_map = dict(zip(kept_columns, output_columns))
	reduced_frame = reduced_frame.rename(columns=rename_map)
	reduced_frame = reduced_frame.loc[:, metadata_columns + output_columns]

	reduction_result = ReductionResult(
		date=str(date),
		hour=int(hour),
		time_column=time_column,
		input_rows=int(frame.shape[0]),
		input_scenarios=int(len(scenario_columns)),
		reduced_scenarios=int(len(kept_columns)),
		kantorovich_distance=kantorovich_distance,
		kept_columns_original=kept_columns,
		output_columns=output_columns,
		probabilities=[float(value) for value in reduced_probabilities],
	)
	return reduced_frame, reduction_result, kept_columns


def reduce_parquet_file(
	input_path: Path,
	output_root: Path,
	target_scenarios: int,
	filter_date: str | None = None,
	filter_hour: int | None = None,
) -> list[ReductionResult]:
	frame = load_parquet_frame(input_path)
	time_column = find_time_column(frame.columns)
	if time_column is None:
		raise ValueError(f"No time column found in {input_path}")

	scenario_columns = get_scenario_columns(frame)
	if not scenario_columns:
		raise ValueError(f"No scenario columns found in {input_path}")

	frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
	time_values = frame[time_column]
	if time_values.isna().all():
		raise ValueError(f"Time column {time_column} could not be parsed in {input_path}")

	output_name = normalize_output_filename(input_path)
	output_parquet = output_root / output_name
	probabilities_dir = output_root / "probabilities" / Path(output_name).stem
	probabilities_dir.mkdir(parents=True, exist_ok=True)

	results: list[ReductionResult] = []
	reduced_frames: list[pd.DataFrame] = []

	valid_mask = time_values.notna()
	if not valid_mask.all():
		frame = frame.loc[valid_mask].copy()
		time_values = time_values.loc[valid_mask]

	date_values = time_values.dt.strftime("%Y-%m-%d")
	hour_values = time_values.dt.hour

	grouped = frame.groupby([date_values, hour_values], sort=True)
	for (date, hour), hour_frame in grouped:
		if hour_frame.empty:
			continue
		if filter_date is not None and str(date) != filter_date:
			continue
		if filter_hour is not None and int(hour) != filter_hour:
			continue

		reduced_frame, result, _ = reduce_hour_frame(
			hour_frame,
			scenario_columns=scenario_columns,
			target_scenarios=target_scenarios,
			date=str(date),
			hour=int(hour),
			time_column=time_column,
		)

		day_dir = probabilities_dir / str(date)
		day_dir.mkdir(parents=True, exist_ok=True)
		output_meta = day_dir / f"hour_{hour:02d}.json"
		with output_meta.open("w", encoding="utf-8") as handle:
			json.dump(asdict(result), handle, indent=2)

		reduced_frames.append(reduced_frame)
		results.append(result)

	if reduced_frames:
		combined_frame = pd.concat(reduced_frames, ignore_index=True)
		if (filter_date is not None or filter_hour is not None) and output_parquet.exists():
			existing_frame = pd.read_parquet(output_parquet)
			combined_frame = pd.concat([existing_frame, combined_frame], ignore_index=True)
			combined_frame = combined_frame.drop_duplicates()
		combined_frame.to_parquet(output_parquet, index=False)

	return results


def discover_input_files(
	input_root: Path,
	output_root: Path | None = None,
	markets: Sequence[str] | None = None,
) -> list[Path]:
	files = []
	output_root_resolved = output_root.resolve() if output_root is not None else None

	for path in input_root.rglob("*.parquet"):
		if not path.is_file():
			continue
		if output_root_resolved is not None:
			try:
				if output_root_resolved in path.resolve().parents:
					continue
			except Exception:
				pass

		if markets and "all" not in markets:
			try:
				relative_path = path.relative_to(input_root)
				if not relative_path.parts or relative_path.parts[0] not in markets:
					continue
			except Exception:
				continue
		files.append(path)

	return sorted(files)


def main() -> None:
	args = parse_args()
	input_root = Path(args.input_root)
	output_root = Path(args.output_root)
	if args.timestamp is not None:
		if args.date is not None or args.hour is not None:
			raise ValueError("--timestamp cannot be combined with --date or --hour")
		args.date, args.hour = parse_timestamp(args.timestamp)
	if args.hour is not None and not 0 <= args.hour <= 23:
		raise ValueError("--hour must be in range 0-23")

	if args.file is not None:
		input_files = [Path(args.file)]
	else:
		input_files = discover_input_files(input_root, output_root, args.markets)

	if not input_files:
		print(f"No parquet files found in {input_root}")
		return

	all_results: list[ReductionResult] = []
	for input_path in input_files:
		print(f"Processing {input_path}")
		results = reduce_parquet_file(
			input_path=input_path,
			output_root=output_root,
			target_scenarios=args.target_scenarios,
			filter_date=args.date,
			filter_hour=args.hour,
		)
		all_results.extend(results)

	summary_rows = [asdict(result) for result in all_results]
	if summary_rows:
		summary_frame = pd.DataFrame(summary_rows)
		preview_frame = summary_frame.sort_values(["date", "hour"]).head(1)
		print(
			preview_frame[
				[
					"date",
					"hour",
					"input_rows",
					"input_scenarios",
					"reduced_scenarios",
					"kantorovich_distance",
				]
			].to_string(index=False)
		)
	elif args.date is not None or args.hour is not None:
		print("No matching date/hour slices found for the filters provided.")


if __name__ == "__main__":
	main()


'''--------
How to run:

To run on one file:
python scenred_backred/backwards_reduction.py --file scenred_backred/updated_data_26/path/to/file.parquet --target-scenarios 20 --output-root scenred_backred/reduced_data_26

Example dayahead:
python scenred_backred/backwards_reduction.py --file scenred_backred/updated_data_26/dayahead/dayahead_forecasts_PT1H.parquet --target-scenarios 20 --output-root scenred_backred/reduced_data_26

Example with date/hour filter:
python scenred_backred/backwards_reduction.py --file scenred_backred/updated_data_26/dayahead/dayahead_forecasts_PT1H.parquet --target-scenarios 20 --output-root scenred_backred/reduced_data_26 --date 2025-01-01 --hour 0

To run on all parquet files under updated_data_26:
python scenred_backred/backwards_reduction.py --input-root scenred_backred/updated_data_26 --target-scenarios 20 --output-root scenred_backred/reduced_data_26

--------'''