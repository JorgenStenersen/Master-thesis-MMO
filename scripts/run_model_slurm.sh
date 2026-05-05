#!/bin/bash
#$ -N ph_bidding
#$ -cwd
#$ -j y
#$ -o ph_bidding_$JOB_ID.out
#$ -pe smp 64
#$ -l h_rt=12:00:00

set -euo pipefail

# ----- Cluster modules -----
if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2024.02 || echo "[WARN] Could not load module Anaconda3/2024.02; trying existing conda installation."
    module load gurobi/13.0 || echo "[WARN] Could not load module gurobi/13.0; assuming Gurobi is available via environment variables."
fi

# Force the cluster Gurobi license for all runs.
export GRB_LICENSE_FILE="/share/apps/gurobi/13.0.0/gurobi_client.lic"

# If conda is not on PATH (e.g., module unavailable), try common install locations.
if ! command -v conda >/dev/null 2>&1; then
    for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
        if [ -x "$c" ]; then
            export PATH="$(dirname "$c"):$PATH"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found. Load the correct module or install conda on the cluster node."
    exit 1
fi

# ----- Conda setup -----
eval "$(conda shell.bash hook)"

# Ensure correct project directory
cd /mnt/beegfs/users/jorgenbs/Kode

# Create or update environment from environment.yml (conda-forge binaries)
if ! conda env list | awk '{print $1}' | grep -qx "bidding_model_env"; then
    echo "Conda environment 'bidding_model_env' not found. Creating it now..."
    conda env create -f environment.yml
else
    echo "Conda environment 'bidding_model_env' already exists. Updating it now..."
    conda env update -f environment.yml --prune
fi

conda activate bidding_model_env

# Basic sanity check: package import + Gurobi version visibility
python -c "import numpy, pandas, pyarrow, fastparquet, gurobipy; print(f'deps ok, gurobipy version={gurobipy.gurobi.version()}')"

# ----- Progressive Hedging runtime config -----
# Override these with environment variables at submit time if needed.
TIME_STR="${TIME_STR:-2025-04-04 08:00:00+00:00}"
N_TOTAL="${N_TOTAL:-20}"
N_PER_BUNDLE="${N_PER_BUNDLE:-3}"
NUM_BUNDLES="${NUM_BUNDLES:-100}"
SEED="${SEED:-30}"
ALPHA="${ALPHA:-100}"
EPSILON="${EPSILON:-1}"
MAX_ITER="${MAX_ITER:-100}"
ADAPTIVE_ALPHA="${ADAPTIVE_ALPHA:-1}"
TAU="${TAU:-2.0}"
MU="${MU:-5.0}"
PH_WORKDIR="${PH_WORKDIR:-ph_sge_runs/$JOB_ID}"

# ----- Local parallelism config (single SGE compute node) -----
TOTAL_CORES="${NSLOTS:-1}"
if [ "$TOTAL_CORES" -lt 1 ]; then
    TOTAL_CORES=1
fi

# Number of bundles solved concurrently on this node.
if [ "$NUM_BUNDLES" -lt "$TOTAL_CORES" ]; then
    CONCURRENT_BUNDLES="$NUM_BUNDLES"
else
    CONCURRENT_BUNDLES="$TOTAL_CORES"
fi

# Per-bundle Gurobi threads = floor(total cores / concurrent bundles), at least 1.
GUROBI_THREADS_PER_BUNDLE=$(( TOTAL_CORES / CONCURRENT_BUNDLES ))
if [ "$GUROBI_THREADS_PER_BUNDLE" -lt 1 ]; then
    GUROBI_THREADS_PER_BUNDLE=1
fi

mkdir -p "$PH_WORKDIR"

echo "Starting local-parallel PH run (SGE): bundles=$NUM_BUNDLES, total_cores=$TOTAL_CORES, concurrent_bundles=$CONCURRENT_BUNDLES, gurobi_threads_per_bundle=$GUROBI_THREADS_PER_BUNDLE, workdir=$PH_WORKDIR"

python -m scripts.ph_slurm_coordinator \
    --time-str "$TIME_STR" \
    --n-total "$N_TOTAL" \
    --n-per-bundle "$N_PER_BUNDLE" \
    --num-bundles "$NUM_BUNDLES" \
    --seed "$SEED" \
    --alpha "$ALPHA" \
    --epsilon "$EPSILON" \
    --max-iter "$MAX_ITER" \
    --adaptive-alpha "$ADAPTIVE_ALPHA" \
    --tau "$TAU" \
    --mu "$MU" \
    --work-dir "$PH_WORKDIR" \
    --max-workers "$CONCURRENT_BUNDLES" \
    --gurobi-threads-per-bundle "$GUROBI_THREADS_PER_BUNDLE"


# How to run
# with default parameters:
# qsub -V scripts/run_model_slurm.sh
# with custom parameters (example):
# qsub -V -pe smp 10 -v TIME_STR="2025-10-09 21:00:00+00:00",N_TOTAL=5,N_PER_BUNDLE=3,NUM_BUNDLES=10,SEED=30,ALPHA=100,EPSILON=1,MAX_ITER=50,ADAPTIVE_ALPHA=1,TAU=2.0,MU=10.0 scripts/run_model_slurm.sh
