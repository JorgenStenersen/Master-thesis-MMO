#!/usr/bin/env python3
"""Evaluate LPHA CM bid activation against reduced n=8 scenario forecasts.

The LPHA model activates a CM bid when the market price is at least the bid
price. For a bidding policy file produced by Progressive Hedging, this script
loads the matching reduced scenario parquets for n=8 and computes the expected
activation probability for the CM_up and CM_down stage-1 bids.

Example:
    python experiments/LPHA_CM_evaluation.py \
        results/12-17/Test 5/8-4608/bidding_policy_pha_2025-12-17_05_00_00_00_00_20260522_153311.json \
        --reduced-root scenred_backred/reduced_data_26 \
        --target-scenarios 8
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MARKET_FILES = {
    "CM_up": "mfrr_cm_up_forecasts.parquet",
    "CM_down": "mfrr_cm_down_forecasts.parquet",
}


@dataclass
class ActivationSummary:
    market: str
    bid_quantity: float
    bid_price: float
    mean_price: float
    min_price: float
    max_price: float
    probability_active: float
    probability_inactive: float
    expected_active_quantity: float
    expected_inactive_quantity: float
    active_scenarios: int
    inactive_scenarios: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the activation probability of LPHA CM bids using reduced "
            "scenario parquets."
        )
    )
    parser.add_argument(
        "policy_files",
        nargs="+",
        help="One or more bidding policy JSON files in the LPHA format.",
    )
    parser.add_argument(
        "--reduced-root",
        default="scenred_backred/reduced_data_26",
        help=(
            "Root folder containing the reduced scenario sets. This can be the "
            "parent folder or the n-specific folder itself."
        ),
    )
    parser.add_argument(
        "--target-scenarios",
        type=int,
        default=8,
        help="Target reduced-scenario count to evaluate.",
    )
    parser.add_argument(
        "--area",
        default="NO3",
        help="Area filter used when selecting the reduced rows.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output file (.json or .csv) for the evaluation summary.",
    )
    return parser.parse_args()


def _parse_time_from_policy_name(policy_path: Path) -> pd.Timestamp:
    match = re.search(r"bidding_policy_pha_(\d{4}-\d{2}-\d{2})_(\d{2})_", policy_path.name)
    if match is None:
        raise ValueError(
            f"Could not infer the evaluation time from policy filename: {policy_path.name}"
        )
    date_str, hour_str = match.groups()
    return pd.to_datetime(f"{date_str} {hour_str}:00:00", utc=True)


def _resolve_reduced_root(reduced_root: Path, target_scenarios: int) -> Path:
    if (reduced_root / "dayahead_forecasts.parquet").exists():
        return reduced_root

    nested = reduced_root / str(target_scenarios)
    if (nested / "dayahead_forecasts.parquet").exists():
        return nested

    raise FileNotFoundError(
        f"Could not find reduced parquets under {reduced_root} or {nested}"
    )


def _reset_if_time_index(df: pd.DataFrame) -> pd.DataFrame:
    index_names: list[str] = []
    if getattr(df.index, "names", None) is not None:
        index_names = [name for name in df.index.names if name is not None]
    elif df.index.name is not None:
        index_names = [df.index.name]

    if any(name in index_names for name in ("time", "prediction_for")):
        return df.reset_index()
    return df


def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Series(pd.to_datetime(series, utc=True, errors="coerce"), index=series.index)

    return pd.Series(pd.to_datetime(series, utc=True, errors="coerce"), index=series.index)


def _detect_scenario_columns(columns: list[str]) -> list[str]:
    scenario_columns: list[str] = []
    for column in columns:
        try:
            int(str(column))
        except Exception:
            continue
        scenario_columns.append(str(column))
    return sorted(scenario_columns, key=lambda value: int(value))


def _load_policy(policy_path: Path) -> dict:
    with policy_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_probability_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    probabilities = np.asarray(payload.get("probabilities", []), dtype=float)
    if probabilities.size == 0:
        raise ValueError(f"No probabilities found in {metadata_path}")

    total_probability = float(probabilities.sum())
    if total_probability <= 0:
        raise ValueError(f"Invalid probability sum in {metadata_path}")
    if not np.isclose(total_probability, 1.0):
        probabilities = probabilities / total_probability

    payload["probabilities"] = probabilities.tolist()
    return payload


def _load_market_slice(
    reduced_root: Path,
    market: str,
    target_time: pd.Timestamp,
    area: str,
) -> tuple[np.ndarray, np.ndarray, dict, pd.Series]:
    parquet_name = MARKET_FILES[market]
    parquet_path = reduced_root / parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing reduced parquet: {parquet_path}")

    metadata_path = (
        reduced_root
        / "probabilities"
        / Path(parquet_name).stem
        / target_time.strftime("%Y-%m-%d")
        / f"hour_{target_time.hour:02d}.json"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing reduced probability metadata: {metadata_path}")

    df = _reset_if_time_index(pd.read_parquet(parquet_path))
    if "prediction_for" not in df.columns:
        raise ValueError(f"No prediction_for column found in {parquet_path}")

    df = df.copy()
    df["prediction_for_dt"] = _ensure_datetime_utc(df["prediction_for"]).dt.floor("h")
    df = df.loc[df["prediction_for_dt"] == target_time]

    if area and "area" in df.columns:
        df = df.loc[df["area"] == area]

    if df.empty:
        raise ValueError(f"No reduced rows found in {parquet_path} for {target_time}")

    if "created_at" in df.columns:
        df = df.sort_values("created_at").head(1)
    else:
        df = df.head(1)

    metadata = _load_probability_metadata(metadata_path)
    scenario_columns = metadata.get("output_columns") or _detect_scenario_columns(list(df.columns))
    if not scenario_columns:
        raise ValueError(f"No scenario columns found in {parquet_path}")

    missing_columns = [column for column in scenario_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Reduced parquet missing columns {missing_columns} in {parquet_path}")

    values = df.loc[:, scenario_columns].to_numpy(dtype=float).flatten()
    probabilities = np.asarray(metadata["probabilities"], dtype=float)
    if values.size != probabilities.size:
        raise ValueError(
            f"Scenario count mismatch in {parquet_path} and {metadata_path} "
            f"(values={values.size}, probabilities={probabilities.size})"
        )

    return values, probabilities, metadata, df.iloc[0]


def _evaluate_market(
    market: str,
    bid_quantity: float,
    bid_price: float,
    reduced_root: Path,
    target_time: pd.Timestamp,
    area: str,
) -> ActivationSummary:
    prices, probabilities, _, _ = _load_market_slice(
        reduced_root=reduced_root,
        market=market,
        target_time=target_time,
        area=area,
    )

    active_mask = prices >= float(bid_price)
    probability_active = float(probabilities[active_mask].sum())
    probability_inactive = 1.0 - probability_active

    return ActivationSummary(
        market=market,
        bid_quantity=float(bid_quantity),
        bid_price=float(bid_price),
        mean_price=float(np.average(prices, weights=probabilities)),
        min_price=float(np.min(prices)),
        max_price=float(np.max(prices)),
        probability_active=probability_active,
        probability_inactive=probability_inactive,
        expected_active_quantity=float(bid_quantity) * probability_active,
        expected_inactive_quantity=float(bid_quantity) * probability_inactive,
        active_scenarios=int(active_mask.sum()),
        inactive_scenarios=int((~active_mask).sum()),
    )


def _evaluate_policy(
    policy_path: Path,
    reduced_root: Path,
    area: str,
) -> dict:
    target_time = _parse_time_from_policy_name(policy_path)
    policy = _load_policy(policy_path)
    stage1_entries = policy.get("stage1", [])

    stage1_by_market = {}
    for entry in stage1_entries:
        market = entry.get("m")
        if market in MARKET_FILES:
            stage1_by_market[market] = entry

    missing_markets = [market for market in MARKET_FILES if market not in stage1_by_market]
    if missing_markets:
        raise ValueError(
            f"Policy file {policy_path} is missing CM stage-1 bids for: {', '.join(missing_markets)}"
        )

    summaries = []
    for market in ("CM_up", "CM_down"):
        entry = stage1_by_market[market]
        summaries.append(
            _evaluate_market(
                market=market,
                bid_quantity=entry.get("x", 0.0),
                bid_price=entry.get("r", 0.0),
                reduced_root=reduced_root,
                target_time=target_time,
                area=area,
            )
        )

    total_bid_quantity = sum(summary.bid_quantity for summary in summaries)
    total_expected_active_quantity = sum(summary.expected_active_quantity for summary in summaries)
    total_expected_inactive_quantity = sum(summary.expected_inactive_quantity for summary in summaries)
    weighted_activation_probability = (
        total_expected_active_quantity / total_bid_quantity if total_bid_quantity > 0 else float("nan")
    )

    return {
        "policy_file": str(policy_path),
        "time": target_time.isoformat(),
        "reduced_root": str(reduced_root),
        "area": area,
        "market_summaries": [asdict(summary) for summary in summaries],
        "aggregate": {
            "total_bid_quantity": float(total_bid_quantity),
            "total_expected_active_quantity": float(total_expected_active_quantity),
            "total_expected_inactive_quantity": float(total_expected_inactive_quantity),
            "quantity_weighted_activation_probability": float(weighted_activation_probability),
            "mean_activation_probability": float(np.mean([summary.probability_active for summary in summaries])),
        },
    }


def _print_result(result: dict) -> None:
    print(f"Policy: {result['policy_file']}")
    print(f"Time:   {result['time']}")
    print(f"Area:   {result['area']}")
    print()

    frame = pd.DataFrame(result["market_summaries"])
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(frame.to_string(index=False, float_format=lambda value: f"{value:0.6f}"))

    aggregate = result["aggregate"]
    print()
    print("Aggregate CM summary")
    print(
        f"  total_bid_quantity                = {aggregate['total_bid_quantity']:.6f}\n"
        f"  total_expected_active_quantity    = {aggregate['total_expected_active_quantity']:.6f}\n"
        f"  total_expected_inactive_quantity  = {aggregate['total_expected_inactive_quantity']:.6f}\n"
        f"  quantity_weighted_activation_prob = {aggregate['quantity_weighted_activation_probability']:.6f}\n"
        f"  mean_activation_probability       = {aggregate['mean_activation_probability']:.6f}"
    )


def _write_output(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".csv":
        rows = []
        for result in results:
            for summary in result["market_summaries"]:
                row = {
                    "policy_file": result["policy_file"],
                    "time": result["time"],
                    "area": result["area"],
                    **summary,
                }
                rows.append(row)
            aggregate = {f"aggregate_{key}": value for key, value in result["aggregate"].items()}
            rows.append({
                "policy_file": result["policy_file"],
                "time": result["time"],
                "area": result["area"],
                "market": "aggregate",
                **aggregate,
            })
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return

    payload: object = results[0] if len(results) == 1 else results
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    reduced_root = _resolve_reduced_root(Path(args.reduced_root), args.target_scenarios)

    results: list[dict] = []
    for policy_file in args.policy_files:
        policy_path = Path(policy_file)
        if not policy_path.exists():
            raise FileNotFoundError(f"Missing policy file: {policy_path}")

        result = _evaluate_policy(policy_path, reduced_root=reduced_root, area=args.area)
        results.append(result)
        _print_result(result)

        if len(args.policy_files) > 1:
            print()

    if args.out is not None:
        _write_output(results, Path(args.out))
        print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()