import src.solvers.extensive_form as extensive_form
import src.solvers.progressive_hedging as progressive_hedging
from scripts.ph_slurm_coordinator import run_distributed_ph
from pathlib import Path
from datetime import datetime
import os


if __name__ == "__main__":

    #mode = "extensive_form"
    mode = "progressive_hedging"

    path = "./input_data_10.csv"
    time_str = "2025-06-12 20:00:00+00:00"
    n = 6
    verbose = True
    seed = 30

    
    if mode == "extensive_form":
        extensive_form.run_model(time_str, n, seed=seed, verbose=verbose)

    # --- Progressive Hedging: solve bundles ---
    n_per_bundle = 2
    num_bundles = 10
    alpha = 100
    epsilon = 5
    adaptive_alpha = True
    tau = 2.0
    mu = 5.0
    gap_pct = 0.01
    max_iter = 100

    if mode == "progressive_hedging":
        total_cores = max(1, os.cpu_count() or 1)
        max_workers = min(num_bundles, total_cores)
        gurobi_threads_per_bundle = max(1, total_cores // max_workers)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = Path("ph_local_runs") / run_id

        run_distributed_ph(
            time_str=time_str,
            n_total=n,
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
            work_dir=work_dir,
            max_workers=max_workers,
            gurobi_threads_per_bundle=gurobi_threads_per_bundle,
        )

        if verbose:
            print(f"[INFO] PH artifacts written under: {work_dir}")
