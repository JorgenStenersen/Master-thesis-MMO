import json
import os
from datetime import datetime
from pathlib import Path

import fastparquet
import numpy as np
import pandas as pd
import pyarrow

from scenred_backred import backwards_reduction

import src.utils as utils

REDUCTION_INPUT_ROOT = Path("scenred_backred") / "updated_data_26"
REDUCTION_OUTPUT_ROOT = Path("scenred_backred") / "reduced_data_26"
REDUCTION_INPUT_FILES = {
    "dayahead_forecasts.parquet": REDUCTION_INPUT_ROOT / "dayahead" / "dayahead_forecasts_PT1H.parquet",
    "imbalance_forecasts.parquet": REDUCTION_INPUT_ROOT / "imbalance" / "imbalance_forecasts_PT1H.parquet",
    "mfrr_cm_down_forecasts.parquet": REDUCTION_INPUT_ROOT / "mfrr_cm_down" / "mfrr_cm_down_forecasts_PT1H.parquet",
    "mfrr_cm_up_forecasts.parquet": REDUCTION_INPUT_ROOT / "mfrr_cm_up" / "mfrr_cm_up_forecasts_PT1H.parquet",
    "mfrr_eam_down_forecasts.parquet": REDUCTION_INPUT_ROOT / "mfrr_eam_down" / "mfrr_eam_down_forecasts_PT1H.parquet",
    "mfrr_eam_up_forecasts.parquet": REDUCTION_INPUT_ROOT / "mfrr_eam_up" / "mfrr_eam_up_forecasts_PT1H.parquet",
    "production_forecasts.parquet": REDUCTION_INPUT_ROOT / "production" / "production_forecasts_PT1H.parquet",
}


def load_parameters_from_csv(path):
    """
    Leser parameters.csv med pandas og returnerer seks lister:
    CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed
    """
    df = pd.read_csv(path)

    CM_up      = df["CM_up"].tolist()
    CM_down    = df["CM_down"].tolist()
    DA         = df["DA"].tolist()
    EAM_up     = df["EAM_up"].tolist()
    EAM_down   = df["EAM_down"].tolist()
    wind_speed = df["wind_speed"].tolist()

    return CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed



def load_expected_values_from_csv(path):
    """
    Leser parameters.csv med pandas og returnerer forventede verdier:
    P_CM_up, P_CM_down, P_DA, P_EAM_up, P_EAM_down, Q_mean
    """
    df = pd.read_csv(path)

    # Forventede (gjennomsnittlige) priser og vind
    P_CM_up    = df["CM_up"].mean()
    P_CM_down  = df["CM_down"].mean()
    P_DA       = df["DA"].mean()
    P_EAM_up   = df["EAM_up"].mean()
    P_EAM_down = df["EAM_down"].mean()
    Q_mean     = df["wind_speed"].mean()   # tilgjengelig produksjonskapasitet

    return P_CM_up, P_CM_down, P_DA, P_EAM_up, P_EAM_down, Q_mean

def _reset_if_time_in_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'time' or 'prediction_for' is in the index (or a MultiIndex),
    reset the index so they become normal columns.
    """
    df = df.copy()

    # Collect index names (works for Index and MultiIndex)
    index_names = []
    if getattr(df.index, "names", None) is not None:
        index_names = [n for n in df.index.names if n is not None]
    elif df.index.name is not None:
        index_names = [df.index.name]

    if any(name in index_names for name in ["time", "prediction_for"]):
        df = df.reset_index()

    return df


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """
    Convert int timestamps (seconds or ms since epoch) to UTC datetime.
    Tries to auto-detect unit based on magnitude.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")

    if pd.api.types.is_object_dtype(series):
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        if not parsed.isna().all():
            return parsed

    s = series.dropna()
    if s.empty:
        # Fallback, assume ms
        return pd.to_datetime(series, unit="ms", utc=True)

    median_abs = float(s.astype("int64").abs().median())
    # crude threshold: >1e11 ~ ms since 1970; <1e11 ~ seconds
    unit = "ms" if median_abs > 1e11 else "s"
    return pd.to_datetime(series, unit=unit, utc=True)


def _to_datetime_preserve_tz(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64tz_dtype(series):
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_object_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if not parsed.isna().all():
            return parsed
    return _to_utc_datetime(series)


def _extract_scenario_columns(frame: pd.DataFrame) -> list[str]:
    scenario_columns: list[str] = []
    for column in frame.columns:
        try:
            int(str(column))
        except Exception:
            continue
        scenario_columns.append(str(column))
    return sorted(scenario_columns, key=lambda value: int(value))


def _load_forecast_row_from_parquet(
    file_path: str | Path,
    target_time: pd.Timestamp,
    area: str,
    park: str,
    scenario_columns: list[str] | None = None,
) -> list[float] | None:
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    df = pd.read_parquet(file_path)
    df = _reset_if_time_in_index(df)

    if "prediction_for" not in df.columns:
        return None

    df = df.copy()
    df["prediction_for_dt"] = _to_datetime_preserve_tz(df["prediction_for"])

    prediction_tz = df["prediction_for_dt"].dt.tz
    if prediction_tz is not None:
        target_match = target_time.tz_convert(prediction_tz)
    else:
        target_match = target_time.tz_convert(None)

    mask = df["prediction_for_dt"] == target_match

    if "area" in df.columns:
        mask &= df["area"] == area
    if "park" in df.columns:
        mask &= df["park"] == park

    df_match = df.loc[mask]
    if df_match.empty:
        dt_series = df["prediction_for_dt"].dropna()
        if not dt_series.empty:
            mask = (
                dt_series.dt.date == target_match.date()
            ) & (
                dt_series.dt.hour == target_match.hour
            )
            df_match = df.loc[mask]

    if df_match.empty and prediction_tz is not None:
        # fallback: match by wall-clock time without timezone conversion
        naive_series = df["prediction_for_dt"].dt.tz_convert(None)
        target_naive = target_time.tz_convert(None)
        mask = naive_series == target_naive
        df_match = df.loc[mask]
    if df_match.empty:
        return None

    if "created_at" in df_match.columns:
        df_match = df_match.sort_values("created_at").head(1)
    else:
        df_match = df_match.head(1)

    if scenario_columns is None:
        scenario_columns = _extract_scenario_columns(df_match)
    else:
        scenario_columns = [col for col in scenario_columns if col in df_match.columns]

    if not scenario_columns:
        return None

    row = df_match.iloc[0]
    return row[scenario_columns].astype(float).tolist()


def load_probabilities_metadata(path: str | Path) -> dict:
    """
    Load scenario-reduction probability metadata JSON and normalize probabilities.

    Expected fields include: date, hour, input_scenarios, reduced_scenarios,
    kept_columns_original, output_columns, probabilities.
    """
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    probabilities = np.asarray(payload.get("probabilities", []), dtype=float)
    if probabilities.size == 0:
        raise ValueError(f"No probabilities found in {metadata_path}")

    total_probability = float(probabilities.sum())
    if not np.isclose(total_probability, 1.0):
        probabilities = probabilities / total_probability

    payload["probabilities"] = probabilities.tolist()
    return payload


def load_probabilities_for_slice(
    base_dir: str | Path,
    stem: str,
    date: str,
    hour: int,
) -> dict:
    """
def load_mmo_data(path):
    base_dir: root containing the probabilities directory.
    stem: subfolder under probabilities (e.g., "dayahead_forecasts").
    date: YYYY-MM-DD string.
    hour: 0-23 integer.
    """
    base_path = Path(base_dir)
    metadata_path = (
        base_path
        / "probabilities"
        / stem
        / date
        / f"hour_{int(hour):02d}.json"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing probabilities metadata: {metadata_path}")

    return load_probabilities_metadata(metadata_path)


def _ensure_reduced_forecast_slice(
    filename: str,
    target_time: pd.Timestamp,
    date_str: str,
    hour: int,
    target_scenarios: int,
    area: str,
    park: str,
) -> tuple[list[float], list[float]]:

    # Ensure outputs are placed under a folder for the given target_scenarios
    target_root = REDUCTION_OUTPUT_ROOT / str(target_scenarios)
    reduced_parquet = target_root / filename

    # Create the target directories
    target_root.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        target_root
        / "probabilities"
        / Path(filename).stem
        / date_str
        / f"hour_{hour:02d}.json"
    )

    metadata: dict | None = None

    input_path = REDUCTION_INPUT_FILES.get(filename)
    if input_path is None or not Path(input_path).exists():
        raise FileNotFoundError(
            f"Missing reduction input parquet for {filename}: {input_path}"
        )

    # Ask the reducer to write into the per-target_scenarios folder
    backwards_reduction.reduce_parquet_file(
        input_path=Path(input_path),
        output_root=target_root,
        target_scenarios=target_scenarios,
        filter_date=date_str,
        filter_hour=hour,
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing probabilities metadata: {metadata_path}")

    metadata = load_probabilities_metadata(metadata_path)
    output_columns = metadata.get("output_columns")
    reduced_values = _load_forecast_row_from_parquet(
        reduced_parquet,
        target_time=target_time,
        area=area,
        park=park,
        scenario_columns=output_columns,
    )

    if reduced_values is None:
        raise ValueError(f"No reduced forecast rows found in {reduced_parquet} for {target_time}")

    probabilities = metadata.get("probabilities")
    if probabilities is None:
        raise ValueError(f"Missing probabilities in {metadata_path}")
    if len(probabilities) != len(reduced_values):
        raise ValueError(
            "Probabilities length does not match reduced scenarios "
            f"in {metadata_path} (probs={len(probabilities)}, values={len(reduced_values)})"
        )

    return reduced_values, probabilities



def load_mmo_data(path):
    df = pd.read_parquet(path)

    print(df.head(6))


def load_parameters_from_parquet(time_str: str, scenarios: int, seed=None, use_reduced_scenarios: bool = True):
    print(f"\nLoading market data for time: {time_str}")
    target_time = pd.to_datetime(time_str, utc=True)
    date_str = target_time.strftime("%Y-%m-%d")
    hour = int(target_time.hour)

    data = {}

    probabilities: dict[str, list[float]] = {}
    if use_reduced_scenarios:
        forecast_files = {
            "mfrr_cm_up_forecasts.parquet": ("mfrr_cm_up_forecasts", "CM_up"),
            "mfrr_cm_down_forecasts.parquet": ("mfrr_cm_down_forecasts", "CM_down"),
            "dayahead_forecasts.parquet": ("dayahead_forecasts", "DA"),
            "mfrr_eam_up_forecasts.parquet": ("mfrr_eam_up_forecasts", "EAM_up"),
            "mfrr_eam_down_forecasts.parquet": ("mfrr_eam_down_forecasts", "EAM_down"),
            "production_forecasts.parquet": ("production_forecasts", "wind_speed"),
        }

        for filename, (var_name, input_key) in forecast_files.items():
            reduced_values, reduced_probs = _ensure_reduced_forecast_slice(
                filename=filename,
                target_time=target_time,
                date_str=date_str,
                hour=hour,
                target_scenarios=scenarios,
                area="NO3",
                park="roan",
            )
            data[var_name] = reduced_values
            probabilities[input_key] = reduced_probs

    # Extract correct lists
    CM_up      = data["mfrr_cm_up_forecasts"]
    CM_down    = data["mfrr_cm_down_forecasts"]
    DA         = data["dayahead_forecasts"]
    EAM_up     = data["mfrr_eam_up_forecasts"]
    EAM_down   = data["mfrr_eam_down_forecasts"]
    wind_speed = data["production_forecasts"]
    
    # Goes through EAM_down and changes the sign of each value
    for i in range(len(EAM_down)):
        EAM_down[i] = -EAM_down[i]

    input_data = {
        "CM_up": CM_up,
        "CM_down": CM_down,
        "DA": DA,
        "EAM_up": EAM_up,
        "EAM_down": EAM_down,
        "wind_speed": wind_speed,
    }
    
    input_data["probabilities"] = probabilities

    return input_data



def get_global_bounds_from_input_data(input_data: dict):
    """
    Henter globale grenseverdier fra input-data dictionary.
    
    Args:
        input_data: Dictionary med:
            - "CM_up": Liste med CM up priser
            - "CM_down": Liste med CM down priser
            - "DA": Liste med day-ahead priser
            - "EAM_up": Liste med EAM up priser
            - "EAM_down": Liste med EAM down priser
            - "wind_speed": Liste med vindhastighetsverdier
    
    Returns:
        Dict med:
        - "Qmax": Høyeste produksjonkapasitet (wind speed)
        - "Pmax": Høyeste pris på tvers av alle markeder
        - "Pmax_per_market": Dict med høyeste pris per marked
                            Nøkler: "CM_up", "CM_down", "DA", "EAM_up", "EAM_down"
    """
    
    # Hent data fra input dictionary
    CM_up = input_data["CM_up"]
    CM_down = input_data["CM_down"]
    DA = input_data["DA"]
    EAM_up = input_data["EAM_up"]
    EAM_down = input_data["EAM_down"]
    wind_speed = input_data["wind_speed"]
    
    # Samle alle priser
    all_prices = []
    Pmax_per_market = {}
    
    # CM markets
    if CM_up:
        cm_up_prices = [float(p) for p in CM_up]
        Pmax_per_market["CM_up"] = max(cm_up_prices)
        all_prices.extend(cm_up_prices)
    
    if CM_down:
        cm_down_prices = [float(p) for p in CM_down]
        Pmax_per_market["CM_down"] = max(cm_down_prices)
        all_prices.extend(cm_down_prices)
    
    # DA market
    if DA:
        da_prices = [float(p) for p in DA]
        Pmax_per_market["DA"] = max(da_prices)
        all_prices.extend(da_prices)
    
    # EAM markets
    if EAM_up:
        eam_up_prices = [float(p) for p in EAM_up]
        Pmax_per_market["EAM_up"] = max(eam_up_prices)
        all_prices.extend(eam_up_prices)
    
    if EAM_down:
        eam_down_prices = [float(p) for p in EAM_down]
        # EAM_down er negativ i dataene, så tar abs for å få maksimal verdi
        Pmax_per_market["EAM_down"] = max(abs(p) for p in eam_down_prices)
        all_prices.extend([abs(p) for p in eam_down_prices])
    
    # Samle alle produksjonsverdier
    Qmax = 0
    if wind_speed:
        wind_speeds = [float(w) for w in wind_speed]
        Qmax = max(wind_speeds)
    
    # Finn høyeste pris på tvers av alle markeder
    Pmax = max(all_prices) if all_prices else 0
    
    global_bounds = {
        "Qmax": Qmax,
        "Pmax": Pmax,
        "Pmax_per_market": Pmax_per_market
    }
    
    print(f"✓ Global bounds computed from input data:")
    print(f"  - Qmax: {Qmax:.4f}")
    print(f"  - Pmax: {Pmax:.4f}")
    print(f"  - Markets with data: {list(Pmax_per_market.keys())}")
    
    return global_bounds


def get_bundle_data(input_data: dict, n_per_bundle: int, seed=None):
    """
    Henter data for én bundle ved å tilfeldig velge n_per_bundle verdier for hver parameter.
    """
    CM_up      = input_data["CM_up"]
    CM_down    = input_data["CM_down"]
    DA         = input_data["DA"]
    EAM_up     = input_data["EAM_up"]
    EAM_down   = input_data["EAM_down"]
    wind_speed = input_data["wind_speed"]

    CM_up_sel, CM_down_sel, DA_sel, EAM_up_sel, EAM_down_sel, wind_speed_sel, picked_scenario_indices = utils.select_possible_realizations_for_bundle(
        n_per_bundle, CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed, seed
    )

    bundle_data = {
        "CM_up": CM_up_sel,
        "CM_down": CM_down_sel,
        "DA": DA_sel,
        "EAM_up": EAM_up_sel,
        "EAM_down": EAM_down_sel,
        "wind_speed": wind_speed_sel,
        # "picked_scenario_indices": picked_scenario_indices --- IGNORE ---
    }

    if "probabilities" in input_data:
        prob_source = input_data["probabilities"]
        if picked_scenario_indices:
            cm_up_indices = [idx[0] for idx in picked_scenario_indices]
            cm_down_indices = [idx[1] for idx in picked_scenario_indices]
            da_indices = [idx[2] for idx in picked_scenario_indices]
            eam_up_indices = [idx[3] for idx in picked_scenario_indices]
            eam_down_indices = [idx[4] for idx in picked_scenario_indices]
            wind_indices = [idx[5] for idx in picked_scenario_indices]

            def pick_probs(values, indices):
                if not values:
                    return [1.0 / len(indices) for _ in indices]
                selected = [values[i] for i in indices]
                total = float(sum(selected))
                if total == 0:
                    return [1.0 / len(selected) for _ in selected]
                return [float(value) / total for value in selected]

            bundle_data["probabilities"] = {
                "CM_up": pick_probs(prob_source.get("CM_up", []), cm_up_indices),
                "CM_down": pick_probs(prob_source.get("CM_down", []), cm_down_indices),
                "DA": pick_probs(prob_source.get("DA", []), da_indices),
                "EAM_up": pick_probs(prob_source.get("EAM_up", []), eam_up_indices),
                "EAM_down": pick_probs(prob_source.get("EAM_down", []), eam_down_indices),
                "wind_speed": pick_probs(prob_source.get("wind_speed", []), wind_indices),
            }
    
    return bundle_data
