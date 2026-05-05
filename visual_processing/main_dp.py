from pathlib import Path

import pandas as pd

from visual_processing.statistics_timestamps import summarize_market_inputs

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "scenred_backred" / "updated_data_26"

TIMESTAMPS_UTC = {
    "T1": "2025-04-06 08:00:00+00:00",
    "T2": "2025-06-12 20:00:00+00:00",
    "T3": "2025-08-22 13:00:00+00:00",
    "T4": "2025-12-17 05:00:00+00:00",
}

PRODUCTION_AVAILABLE_FOR = "dayahead"

PRICE_FORECAST_FILES = {
    "prices_cm_up": DATA_DIR / "mfrr_cm_up" / "mfrr_cm_up_forecasts_PT1H.parquet",
    "prices_cm_down": DATA_DIR / "mfrr_cm_down" / "mfrr_cm_down_forecasts_PT1H.parquet",
    "prices_dam": DATA_DIR / "dayahead" / "dayahead_forecasts_PT1H.parquet",
    "prices_eam_up": DATA_DIR / "mfrr_eam_up" / "mfrr_eam_up_forecasts_PT1H.parquet",
    "prices_eam_down": DATA_DIR / "mfrr_eam_down" / "mfrr_eam_down_forecasts_PT1H.parquet",
}

PRODUCTION_FORECAST_FILE = (
    DATA_DIR / "production" / "production_forecasts_PT1H.parquet"
)


def _scenario_columns(columns):
    scenario_cols = [col for col in columns if str(col).isdigit()]
    return sorted(scenario_cols, key=lambda c: int(c))


def _read_price_forecast(path, timestamp_utc, area="NO3"):
    df = pd.read_parquet(path)
    ts = pd.Timestamp(timestamp_utc)

    if "prediction_for" in df.columns:
        df = df[df["prediction_for"] == ts]
    elif df.index.name == "time":
        df = df.loc[[ts]]
    else:
        raise ValueError(f"No timestamp column found in {path}")

    if "area" in df.columns:
        df = df[df["area"] == area]

    if df.empty:
        raise ValueError(f"No data for {timestamp_utc} in {path}")

    cols = _scenario_columns(df.columns)
    if not cols:
        raise ValueError(f"No scenario columns in {path}")

    row = df.iloc[0]
    return [float(row[col]) for col in cols]


def _read_production_forecast(path, timestamp_utc, available_for):
    df = pd.read_parquet(path)
    ts = pd.Timestamp(timestamp_utc)

    df = df[df["prediction_for"] == ts]
    df = df[df["available_for"] == available_for]

    if df.empty:
        raise ValueError(
            f"No production data for {timestamp_utc} ({available_for}) in {path}"
        )

    if "created_at" in df.columns:
        df = df.sort_values("created_at")

    cols = _scenario_columns(df.columns)
    if not cols:
        raise ValueError(f"No scenario columns in {path}")

    row = df.iloc[-1]
    return [float(row[col]) for col in cols]


timestamp_data = {}
for label, ts in TIMESTAMPS_UTC.items():
    timestamp_data[label] = {
        key: _read_price_forecast(path, ts)
        for key, path in PRICE_FORECAST_FILES.items()
    }
    timestamp_data[label]["forecasted_power"] = _read_production_forecast(
        PRODUCTION_FORECAST_FILE, ts, PRODUCTION_AVAILABLE_FOR
    )


prices_cm_up_t1 = timestamp_data["T1"]["prices_cm_up"]
prices_cm_down_t1 = timestamp_data["T1"]["prices_cm_down"]
prices_dam_t1 = timestamp_data["T1"]["prices_dam"]
prices_eam_up_t1 = timestamp_data["T1"]["prices_eam_up"]
prices_eam_down_t1 = timestamp_data["T1"]["prices_eam_down"]
forecasted_power_t1 = timestamp_data["T1"]["forecasted_power"]

prices_cm_up_t2 = timestamp_data["T2"]["prices_cm_up"]
prices_cm_down_t2 = timestamp_data["T2"]["prices_cm_down"]
prices_dam_t2 = timestamp_data["T2"]["prices_dam"]
prices_eam_up_t2 = timestamp_data["T2"]["prices_eam_up"]
prices_eam_down_t2 = timestamp_data["T2"]["prices_eam_down"]
forecasted_power_t2 = timestamp_data["T2"]["forecasted_power"]

prices_cm_up_t3 = timestamp_data["T3"]["prices_cm_up"]
prices_cm_down_t3 = timestamp_data["T3"]["prices_cm_down"]
prices_dam_t3 = timestamp_data["T3"]["prices_dam"]
prices_eam_up_t3 = timestamp_data["T3"]["prices_eam_up"]
prices_eam_down_t3 = timestamp_data["T3"]["prices_eam_down"]
forecasted_power_t3 = timestamp_data["T3"]["forecasted_power"]

prices_cm_up_t4 = timestamp_data["T4"]["prices_cm_up"]
prices_cm_down_t4 = timestamp_data["T4"]["prices_cm_down"]
prices_dam_t4 = timestamp_data["T4"]["prices_dam"]
prices_eam_up_t4 = timestamp_data["T4"]["prices_eam_up"]
prices_eam_down_t4 = timestamp_data["T4"]["prices_eam_down"]
forecasted_power_t4 = timestamp_data["T4"]["forecasted_power"]


def summarize_timestamp(label):
    data = timestamp_data[label]
    summarize_market_inputs(
        data["prices_cm_up"],
        data["prices_cm_down"],
        data["prices_dam"],
        data["prices_eam_up"],
        data["prices_eam_down"],
        data["forecasted_power"],
    )


if __name__ == "__main__":
    summarize_timestamp("T1")
