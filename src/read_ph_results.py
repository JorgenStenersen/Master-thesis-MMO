import argparse
import pickle
from pathlib import Path


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _summarize_final_state(final_state: dict) -> None:
    status = final_state.get("status")
    iterations = final_state.get("iterations")
    gap = final_state.get("gap")
    alpha = final_state.get("alpha")

    print("Final state summary:")
    print(f"  status: {status}")
    print(f"  iterations: {iterations}")
    print(f"  gap: {gap}")
    print(f"  alpha: {alpha}")

    consensus = final_state.get("consensus", {})
    for stage_name in ("stage1", "stage2", "stage3"):
        stage = consensus.get(stage_name, {})
        print(f"  {stage_name}: {len(stage)} decision(s)")


def _summarize_iter_results(iter_dir: Path) -> None:
    results_dir = iter_dir / "results"
    if not results_dir.exists():
        print(f"No results directory found in {iter_dir}")
        return

    result_files = sorted(results_dir.glob("bundle_*.pkl"))
    print(f"Iteration {iter_dir.name}: {len(result_files)} bundle result(s)")

    solved = 0
    objectives = []
    for path in result_files:
        res = _load_pickle(path)
        if res is None:
            continue
        solved += 1
        obj = res.get("objective")
        if obj is not None:
            objectives.append(obj)

    print(f"  solved bundles: {solved}/{len(result_files)}")
    if objectives:
        print(f"  augmented sobjective min/max: {min(objectives):.4f}/{max(objectives):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read PH output .pkl files.")
    parser.add_argument(
        "work_dir",
        type=str,
        help="Path to PH output folder (e.g., ph_sge_runs/1234 or ph_local_runs/<ts>).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Optional iteration index to summarize (e.g., 0).",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        raise SystemExit(f"Work dir not found: {work_dir}")

    final_state_path = work_dir / "final_state.pkl"
    if final_state_path.exists():
        final_state = _load_pickle(final_state_path)
        _summarize_final_state(final_state)
    else:
        print(f"No final_state.pkl found in {work_dir}")

    if args.iteration is not None:
        iter_dir = work_dir / f"iter_{args.iteration:03d}"
        _summarize_iter_results(iter_dir)


if __name__ == "__main__":
    main()

'''
How to run:
python -m src.read_ph_results ph_sge_runs/5434
python -m src.read_ph_results ph_sge_runs/5434 --iteration 0
'''