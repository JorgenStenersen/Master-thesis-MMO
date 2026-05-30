import src.solvers.extensive_form as extensive_form
import src.solvers.progressive_hedging as progressive_hedging
from scripts.ph_plots import plot_ph_boxplot
from scripts.ph_slurm_coordinator import run_distributed_ph
from pathlib import Path
from datetime import datetime
import os


if __name__ == "__main__":

    mode = "extensive_form"
    #mode = "progressive_hedging"

    path = "./input_data_10.csv"
    time_str = "2025-12-17 05:00:00+00:00"
    n = 8
    verbose = True
    seed = 30

    
    if mode == "extensive_form":
        extensive_form.run_model(time_str, n, seed=seed, verbose=verbose)

    # --- Progressive Hedging: solve bundles ---
    n_per_bundle = 1
    D = 9
    num_bundles = round(D * n**3 / n_per_bundle**3)
    alpha = 100
    epsilon = 0.01
    adaptive_alpha = True
    tau = 2.0
    mu = 10.0
    gap_pct = 0.01
    max_iter = 100

    if mode == "progressive_hedging":
        total_cores = max(1, os.cpu_count() or 1)
        max_workers = min(num_bundles, total_cores)
        gurobi_threads_per_bundle = max(1, total_cores // max_workers)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = Path("ph_local_runs") / run_id

        # Run PH locally (single-process coordinator) and collect results
        B, results, consensus, W_shadow = progressive_hedging.run_progressive_hedging(
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
            bidding_output_dir=work_dir,
            verbose=verbose,
        )

        # Create boxplots of decisions for x and r and save under the PH run directory
        try:
            png_x = work_dir / "ph_consensus_boxplot_x.png"
            plot_ph_boxplot(results, consensus, output_path=png_x, var="x")
            if verbose:
                print(f"[INFO] PH consensus boxplot (x) written to: {png_x}")

            png_r = work_dir / "ph_consensus_boxplot_r.png"
            plot_ph_boxplot(results, consensus, output_path=png_r, var="r")
            if verbose:
                print(f"[INFO] PH consensus boxplot (r) written to: {png_r}")
        except Exception as e:
            print(f"[WARNING] Could not create PH consensus boxplots: {e}")
