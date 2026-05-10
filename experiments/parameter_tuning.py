import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

try:
    import optuna
except Exception as e:
    optuna = None

from scripts.ph_slurm_coordinator import run_distributed_ph


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_trial_and_get_elapsed(time_str: str, n_total: int, n_per_bundle: int, num_bundles: int,
                              seed: int, alpha: float, epsilon: float, max_iter: int,
                              gap_pct: float, adaptive_alpha: bool, tau: float, mu: float,
                              base_work_dir: Path, max_workers: int, gurobi_threads_per_bundle: int) -> float:
    """Run a single PH experiment and return total elapsed seconds from run_summary.json.

    The function creates a unique work_dir for the trial so runs don't overwrite each other.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trial_dir = base_work_dir / f"trial_{ts}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
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
            work_dir=trial_dir,
            max_workers=max_workers,
            gurobi_threads_per_bundle=gurobi_threads_per_bundle,
        )

        # read run_summary.json
        summary_path = trial_dir / "run_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"run_summary.json not found in {trial_dir}")
        with summary_path.open("r", encoding="utf-8") as f:
            run_summary = json.load(f)
        elapsed = run_summary.get("total_elapsed_seconds")
        if elapsed is None:
            # fallback: compute from iteration summaries
            elapsed = float(run_summary.get("total_elapsed_seconds", float("inf")))
        return float(elapsed)
    except Exception:
        # on error, write a traceback file for debugging and return large value
        err_path = trial_dir / "error.txt"
        with err_path.open("w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return float("inf")


def objective_optuna(trial: "optuna.trial.Trial", args) -> float:
    # Suggest hyperparameters
    alpha = trial.suggest_float("alpha", 1, 200, log=True)
    tau = trial.suggest_float("tau", 1.1, 10, log=True)
    mu = trial.suggest_float("mu", 1.1, 10, log=True)

    # Run trial
    start = time.perf_counter()
    elapsed = run_trial_and_get_elapsed(
        time_str=args.time_str,
        n_total=args.n_total,
        n_per_bundle=args.n_per_bundle,
        num_bundles=args.num_bundles,
        seed=args.seed,
        alpha=alpha,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        gap_pct=args.gap_pct,
        adaptive_alpha=True,
        tau=tau,
        mu=mu,
        base_work_dir=Path(args.work_dir) / "hyperopt",
        max_workers=args.max_workers,
        gurobi_threads_per_bundle=args.gurobi_threads_per_bundle,
    )
    end = time.perf_counter()

    # optuna minimizes the returned value
    # Use reported elapsed if available, otherwise wall time
    if elapsed == float("inf"):
        return end - start
    return elapsed


def main(cli_args: dict | None = None):
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for PH (Optuna)")
    parser.add_argument("--time-str", type=str, required=True)
    parser.add_argument("--n-total", type=int, required=True)
    parser.add_argument("--n-per-bundle", type=int, required=True)
    parser.add_argument("--num-bundles", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--gap-pct", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--work-dir", type=str, default="ph_hyperopt_runs")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="ph_opt")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--gurobi-threads-per-bundle", type=int, default=1)

    if cli_args is None:
        args = parser.parse_args()
    else:
        # build args Namespace using parser defaults, then overwrite with provided cli_args
        defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
        merged = defaults.copy()
        # map dashed arg names to dest keys (argparse already uses dest names like time_str)
        merged.update(cli_args)
        args = argparse.Namespace(**merged)

    if optuna is None:
        print("Optuna is not installed. Please install optuna (see requirements.txt) to run this script.")
        return

    study = optuna.create_study(direction="minimize", study_name=args.study_name)
    func = lambda t: objective_optuna(t, args)
    print(f"Starting optimization: study={args.study_name}, n_trials={args.n_trials}")
    study.optimize(func, n_trials=args.n_trials, n_jobs=args.n_jobs)

    best = study.best_trial
    print("Optimization finished")
    print(f"Best value (seconds): {best.value}")
    print(f"Best params: {best.params}")

    # Save summary
    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "study_name": args.study_name,
        "datetime": _now_iso(),
        "n_trials": args.n_trials,
        "best_value": best.value,
        "best_params": best.params,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    with (out_dir / f"optuna_summary_{args.study_name}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Default run settings (can be overridden by passing a dict to main)
    cli_defaults = {
        "time_str": "2025-06-12 20:00:00+00:00",
        "n_total": 6,
        "n_per_bundle": 1,
        "num_bundles": 1080,
        "seed": 1,
        "epsilon": 1e-2,
        "max_iter": 200,
        "work_dir": "ph_hyperopt_runs",
        "n_trials": 50,
        "n_jobs": 1,
        "study_name": "ph_opt",
        "max_workers": 2,
        "gurobi_threads_per_bundle": 1,
    }

    # Call main with the hard-coded defaults (keeps CLI parsing available if needed)
    main(cli_defaults)

#How to run:
# python experiments/parameter_tuning.py ^
#  --time-str "2025-04-06 08:00:00+00:00" ^
#  --n-total 10 --n-per-bundle 2 --num-bundles 5 ^
#  --n-trials 20 --max-workers 2 --gurobi-threads-per-bundle 1 