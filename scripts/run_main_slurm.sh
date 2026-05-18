#!/bin/bash
#$ -N main_bidding
#$ -cwd
#$ -j y
#$ -o main_bidding_$JOB_ID.out
#$ -pe smp 64
#$ -l h_rt=04:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1

# ----- Paths and config -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="/mnt/beegfs/users/jorgenbs/Kode"

_is_repo_root() {
    [ -f "$1/environment.yml" ] && [ -d "$1/scripts" ]
}

REPO_ROOT_CANDIDATES=(
    "${REPO_ROOT:-}"
    "${SGE_O_WORKDIR:-}"
    "$REPO_ROOT_DEFAULT"
    "$(cd "$SCRIPT_DIR/.." && pwd)"
    "${PWD:-}"
)

REPO_ROOT=""
for candidate in "${REPO_ROOT_CANDIDATES[@]}"; do
    if [ -n "$candidate" ] && _is_repo_root "$candidate"; then
        REPO_ROOT="$candidate"
        break
    fi
done

if [ -z "$REPO_ROOT" ]; then
    echo "[ERROR] Repo root not found. Checked: ${REPO_ROOT_CANDIDATES[*]}"
    exit 1
fi
ENV_NAME="${ENV_NAME:-bidding_model_env}"
AUTO_UPDATE_ENV="${AUTO_UPDATE_ENV:-1}"
SKIP_ENV_UPDATE_ON_ARRAY="${SKIP_ENV_UPDATE_ON_ARRAY:-1}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/slurm}"
RUN_MODE="${RUN_MODE:-main}"

# ----- Find conda -----
CONDA_EXE=""
if [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
elif [ -x "$HOME/anaconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/anaconda3/bin/conda"
fi

if [ -z "$CONDA_EXE" ] && command -v module >/dev/null 2>&1; then
    module load Anaconda3/2024.02 || true
    CONDA_EXE="$(command -v conda || true)"
fi

if [ -z "$CONDA_EXE" ]; then
    echo "[ERROR] conda not found. Install Miniconda/Anaconda or load a module."
    exit 1
fi

# ----- Project directory -----
cd "$REPO_ROOT"

# ----- Logging -----
mkdir -p "$LOG_DIR"
JOB_ID="${JOB_ID:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"
LOG_FILE="$LOG_DIR/main_bidding_${JOB_ID}.log"
echo "[INFO] Writing output to: $LOG_FILE"
exec >"$LOG_FILE" 2>&1

on_terminate() {
    echo "[ERROR] Job received termination signal at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    echo "[ERROR] Likely scheduler timeout or external kill."
}
trap on_terminate TERM INT

# ----- Gurobi license setup -----
if command -v module >/dev/null 2>&1; then
    module load gurobi/13.0 || true
fi
export GRB_LICENSE_FILE="/share/apps/gurobi/13.0.0/gurobi_client.lic"

# ----- Create or update environment (optional) -----
if [ "$AUTO_UPDATE_ENV" = "1" ]; then
    if [ "$SKIP_ENV_UPDATE_ON_ARRAY" = "1" ] && [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ] && [ "${SGE_TASK_ID}" != "1" ]; then
        echo "Skipping env update for array task ${SGE_TASK_ID}."
    else
    if ! "$CONDA_EXE" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' not found. Creating it now..."
        "$CONDA_EXE" env create -f environment.yml
    else
        echo "Conda environment '$ENV_NAME' already exists. Updating it now..."
        "$CONDA_EXE" env update -f environment.yml --prune
    fi
    fi
else
    echo "Skipping env update (AUTO_UPDATE_ENV=$AUTO_UPDATE_ENV)."
fi

# ----- Sanity check -----
"$CONDA_EXE" run -n "$ENV_NAME" python -c "import numpy, pandas, pyarrow, fastparquet, gurobipy; print('deps ok')"

# ----- Gurobi license diagnostics -----
"$CONDA_EXE" run -n "$ENV_NAME" python - <<'PY'
import os
print("GRB_LICENSE_FILE:", os.environ.get("GRB_LICENSE_FILE"))
print("GUROBI_HOME:", os.environ.get("GUROBI_HOME"))
PY
echo "[INFO] GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-<unset>}"
echo "[INFO] GUROBI_HOME=${GUROBI_HOME:-<unset>}"
GUROBI_HOME_SAFE="${GUROBI_HOME:-}"
if [ -n "$GUROBI_HOME_SAFE" ]; then
    ls -l "$HOME/.gurobi/gurobi.lic" "$HOME/gurobi.lic" "./gurobi.lic" "$GUROBI_HOME_SAFE/gurobi.lic" 2>/dev/null || true
else
    ls -l "$HOME/.gurobi/gurobi.lic" "$HOME/gurobi.lic" "./gurobi.lic" 2>/dev/null || true
fi

if [ -n "$GUROBI_HOME_SAFE" ] && [ -x "$GUROBI_HOME_SAFE/bin/gurobi_cl" ]; then
    "$GUROBI_HOME_SAFE/bin/gurobi_cl" --license || true
elif command -v gurobi_cl >/dev/null 2>&1; then
    gurobi_cl --license || true
fi

# ----- Run selection -----
if [ "$RUN_MODE" = "bench" ]; then
    BENCH_RUN_TYPE="${BENCH_RUN_TYPE:-both}"
    PH_BACKEND="${PH_BACKEND:-coordinator}"
    RESULTS_ROOT="${RESULTS_ROOT:-results}"
    EXTENSIVE_N="${EXTENSIVE_N:-}"
    BENCH_COMBOS="${BENCH_COMBOS:-}"
    BENCH_OUT="${BENCH_OUT:-}"
    PH_WORK_ROOT="${PH_WORK_ROOT:-}"

    TIME_STR_1="${TIME_STR_1:-}"
    TIME_STR_2="${TIME_STR_2:-}"
    TIME_STR_3="${TIME_STR_3:-}"
    TIME_STR_4="${TIME_STR_4:-}"

    if [ -z "$TIME_STR_1" ] || [ -z "$TIME_STR_2" ] || [ -z "$TIME_STR_3" ] || [ -z "$TIME_STR_4" ]; then
        echo "[ERROR] TIME_STR_1..TIME_STR_4 must be set for bench mode."
        exit 1
    fi

    if [ "$BENCH_RUN_TYPE" != "ph" ] && [ -z "$EXTENSIVE_N" ]; then
        echo "[ERROR] EXTENSIVE_N must be set for extensive or both modes."
        exit 1
    fi

    if [ "$BENCH_RUN_TYPE" != "extensive" ] && [ -z "$BENCH_COMBOS" ]; then
        echo "[ERROR] BENCH_COMBOS must be set for PH or both modes."
        exit 1
    fi

    # ----- Local parallelism config (single SGE compute node) -----
    TOTAL_CORES="${NSLOTS:-1}"
    if [ "$TOTAL_CORES" -lt 1 ]; then
        TOTAL_CORES=1
    fi

    PH_MAX_WORKERS="${PH_MAX_WORKERS:-$TOTAL_CORES}"
    GUROBI_THREADS_PER_BUNDLE="${GUROBI_THREADS_PER_BUNDLE:-1}"

    BENCH_COMBO_INDEX="${BENCH_COMBO_INDEX:-}"
    if [ -z "$BENCH_COMBO_INDEX" ] && [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
        BENCH_COMBO_INDEX=$((SGE_TASK_ID - 1))
    fi

    BENCH_ARGS=(
        --time-str "$TIME_STR_1"
        --time-str "$TIME_STR_2"
        --time-str "$TIME_STR_3"
        --time-str "$TIME_STR_4"
        --run-type "$BENCH_RUN_TYPE"
        --ph-backend "$PH_BACKEND"
        --results-root "$RESULTS_ROOT"
        --alpha "${ALPHA:-100}"
        --epsilon "${EPSILON:-20}"
        --max-iter "${MAX_ITER:-100}"
        --adaptive-alpha "${ADAPTIVE_ALPHA:-1}"
        --tau "${TAU:-2.0}"
        --mu "${MU:-10.0}"
        --seed "${SEED:-30}"
        --max-workers "$PH_MAX_WORKERS"
        --gurobi-threads-per-bundle "$GUROBI_THREADS_PER_BUNDLE"
    )

    if [ -n "$EXTENSIVE_N" ]; then
        BENCH_ARGS+=(--extensive-n "$EXTENSIVE_N")
    fi

    if [ -n "$BENCH_COMBOS" ]; then
        combo_sep=";"
        if [[ "$BENCH_COMBOS" == *","* && "$BENCH_COMBOS" != *";"* ]]; then
            combo_sep=","
        fi
        IFS="$combo_sep" read -r -a combo_list <<< "$BENCH_COMBOS"
        for combo in "${combo_list[@]}"; do
            BENCH_ARGS+=(--combo "$combo")
        done
    fi

    if [ -n "$BENCH_OUT" ]; then
        BENCH_ARGS+=(--out "$BENCH_OUT")
    fi

    if [ -n "$PH_WORK_ROOT" ]; then
        BENCH_ARGS+=(--ph-work-root "$PH_WORK_ROOT")
    fi

    if [ -n "$BENCH_COMBO_INDEX" ]; then
        BENCH_ARGS+=(--combo-index "$BENCH_COMBO_INDEX")
    fi

    echo "[INFO] Starting benchmark run (mode=$BENCH_RUN_TYPE, backend=$PH_BACKEND)"
    PYTHONPATH="$REPO_ROOT" "$CONDA_EXE" run -n "$ENV_NAME" python experiments/run_ph_bench.py "${BENCH_ARGS[@]}"
else
    # ----- Run main.py -----
    echo "[INFO] Launching python -m scripts.main"
    if "$CONDA_EXE" run -n "$ENV_NAME" python -u -m scripts.main; then
        echo "[INFO] python -m scripts.main finished with exit code 0"
    else
        status=$?
        echo "[ERROR] python -m scripts.main failed with exit code $status"
        exit "$status"
    fi
fi

# Bench job-array examples (SGE):
# Extensive (single job):
# qsub -V -pe smp 8 -v RUN_MODE=bench,BENCH_RUN_TYPE=extensive,EXTENSIVE_N=20,TIME_STR_1="...",TIME_STR_2="...",TIME_STR_3="...",TIME_STR_4="..." scripts/run_main_slurm.sh
# PH (array, one combo per task):
# qsub -V -pe smp 64 -t 1-6 -v RUN_MODE=bench,BENCH_RUN_TYPE=ph,BENCH_COMBOS="8:1:768;8:1:1024;8:1:1280;8:1:1536;8:1:1792;8:1:2048",TIME_STR_1="2025-04-04 08:00:00+00:00",TIME_STR_2="2025-06-10 20:00:00+00:00",TIME_STR_3="2025-08-20 13:00:00+00:00",TIME_STR_4="2025-12-15 05:00:00+00:00",RESULTS_ROOT="results" scripts/run_main_slurm.sh
# PH (array, with cache cleaning to avoid contention):
# qsub -V -pe smp 64 -t 1-6 -v RUN_MODE=bench,BENCH_RUN_TYPE=ph,BENCH_COMBOS="7:1:515;7:1:686;7:1:856;7:1:1029;7:1:1201;7:1:1372",TIME_STR_1="2025-04-04 08:00:00+00:00",TIME_STR_2="2025-06-10 20:00:00+00:00",TIME_STR_3="2025-08-22 13:00:00+00:00",TIME_STR_4="2025-12-17 05:00:00+00:00",RESULTS_ROOT="results",AUTO_UPDATE_ENV=0,CONDA_PKGS_DIRS="/tmp/$JOB_ID/conda_pkgs" scripts/run_main_slurm.sh
# Pure main, no benchmark (requires edit for n in main.py):
# qsub -V -pe smp 8 scripts/run_main_slurm.sh