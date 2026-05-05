from gurobipy import GRB

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import re
import os

import src.read as read
import src.utils as utils
from src.model import build_model, initialize_run
import src.tree as tree


def _sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def _write_extensive_results(model_container, time_str, n, seed, status=None, error=None) -> Path:
    model = model_container.model
    vars_map = model_container.vars
    sets_map = model_container.sets

    cm_decisions = []
    objective_value = None
    runtime_seconds = None
    gap = None

    status_value = status
    if status_value is None:
        status_value = getattr(model, "Status", None)

    if status_value == GRB.OPTIMAL:
        x = vars_map["x"]
        r = vars_map["r"]
        a = vars_map["a"]
        delta = vars_map["delta"]

        U = utils.sort_nodes(sets_map["U"])
        M_u = sorted(sets_map["M_u"])

        for u in U:
            for m in M_u:
                cm_decisions.append(
                    {
                        "u": str(u),
                        "market": str(m),
                        "x": float(x[m, u].X),
                        "r": float(r[m, u].X),
                        "a": float(a[m, u].X),
                        "delta": int(round(delta[m, u].X)),
                        "d": None,
                    }
                )

        objective_value = float(model.ObjVal)
        runtime_seconds = float(model.Runtime)

        if getattr(model, "IsMIP", 0):
            try:
                gap = float(model.MIPGap)
            except AttributeError:
                gap = None

    results = {
        "time_str": str(time_str),
        "n": int(n),
        "seed": None if seed is None else int(seed),
        "status": -1 if status_value is None else int(status_value),
        "objective_value": objective_value,
        "runtime_seconds": runtime_seconds,
        "optimality_gap": gap,
        "error": error,
        "cm_decisions": cm_decisions,
    }

    results_root = Path("results") / "extensive"
    results_root.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_time = _sanitize_for_filename(str(time_str))
    output_path = results_root / f"extensive_{safe_time}_n{n}_{run_stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return output_path


def _write_extensive_checkpoint(time_str, n, seed, phase, message=None) -> Path:
    results_root = Path("results") / "extensive"
    results_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "time_str": str(time_str),
        "n": int(n),
        "seed": None if seed is None else int(seed),
        "phase": phase,
        "message": message,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }

    checkpoint_path = results_root / "extensive_last_run_status.json"
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return checkpoint_path


def run_model(time_str, n, seed=None, verbose=True):
    run_started = time.perf_counter()
    checkpoint_path = _write_extensive_checkpoint(
        time_str,
        n,
        seed,
        phase="starting",
        message="Extensive-form run initialized",
    )
    print(f"[INFO] Extensive checkpoint: {checkpoint_path}")
    print(f"[INFO] Loading input data for time_str={time_str}, n={n}, seed={seed}")
    input_data = read.load_parameters_from_parquet(time_str, n, seed)
    _write_extensive_checkpoint(time_str, n, seed, phase="input_data_loaded")

    print("[INFO] Building scenario tree...")
    scenario_tree = tree.build_scenario_tree(input_data)
    _write_extensive_checkpoint(time_str, n, seed, phase="scenario_tree_built")

    print("[INFO] Computing global bounds from input data...")
    global_bounds = read.get_global_bounds_from_input_data(input_data)
    _write_extensive_checkpoint(time_str, n, seed, phase="global_bounds_ready")


    # Bygg modell
    print("[INFO] Building optimization model...")
    model_container = build_model(scenario_tree, global_bounds)
    _write_extensive_checkpoint(time_str, n, seed, phase="model_built")

    log_dir = Path("results") / "extensive" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_time = _sanitize_for_filename(str(time_str))
    gurobi_log_path = log_dir / f"gurobi_extensive_{safe_time}_n{n}_{run_stamp}.log"
    try:
        model_container.model.setParam("LogFile", str(gurobi_log_path))
        print(f"[INFO] Gurobi LogFile set to: {gurobi_log_path}")
    except Exception as exc:
        print(f"[WARN] Failed to set Gurobi LogFile: {exc}")

    print(f"[INFO] Starting extensive form run: time_str={time_str}, n={n}, seed={seed}")
    _write_extensive_checkpoint(
        time_str,
        n,
        seed,
        phase="optimizing",
        message=f"Gurobi log: {gurobi_log_path}",
    )

    error = None
    status = None
    results_path = None
    try:
        # --- OPTIMIZE MODEL ---
        model_container.model.optimize()
        status = getattr(model_container.model, "Status", None)

        if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            if os.environ.get("COMPUTE_IIS") == "1":
                try:
                    # Capture minimal IIS info to debug infeasibility.
                    model_container.model.computeIIS()
                    model_container.model.write("infeasible.ilp")
                    print("[ERROR] Model infeasible or unbounded. IIS written to infeasible.ilp")
                except Exception as exc:
                    print(f"[WARN] IIS computation failed: {exc}")
            else:
                print("[ERROR] Model infeasible or unbounded. Set COMPUTE_IIS=1 to compute IIS.")
        else:
            print(f"Model optimized in {model_container.model.Runtime:.2f} seconds.")

        if verbose and status == GRB.OPTIMAL:
            utils.print_results(model_container)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            results_path = _write_extensive_results(
                model_container,
                time_str,
                n,
                seed,
                status=status,
                error=error,
            )
            print(f"[INFO] Extensive results written to: {results_path}")
            _write_extensive_checkpoint(
                time_str,
                n,
                seed,
                phase="finished",
                message=f"Results written to {results_path}",
            )
        except Exception as exc:
            print(f"[ERROR] Failed to write extensive results: {exc}")
            _write_extensive_checkpoint(
                time_str,
                n,
                seed,
                phase="failed_to_write_results",
                message=f"{type(exc).__name__}: {exc}",
            )
        if status is not None:
            print(f"[INFO] Extensive run finished with status: {status}")
        print(f"[INFO] Extensive run wall time: {time.perf_counter() - run_started:.2f}s")






