#!/bin/bash
#$ -cwd
#$ -j y
#$ -N ph_combo_sweep

set -euo pipefail

# Usage (example):
#   qsub -pe smp 64 -v TIME_STR="2025-08-22 13:00:00+00:00",MAX_WORKERS=64,COMBOS="4:1:576 5:1:1125 7:1:3087 9:1:6561" scripts/qsub_ph_combo_sweep.sh
#
# Optional environment variables:
#   TIME_STR                 Timestamp to run (default: script default)
#   MAX_WORKERS              Max worker processes (default: NSLOTS)
#   GUROBI_THREADS_PER_BUNDLE Threads per bundle (default: floor(NSLOTS/MAX_WORKERS))
#   COMBOS                   Space-separated combos, e.g. "4:1:576 6:1:1944"
#   RESULTS_ROOT             Output root for CSV/results
#   PH_WORK_ROOT             Work dir root for PH artifacts
#   OUT                      CSV output path
#   EXTRA_ARGS               Extra args to append verbatim
#   VENV_ACTIVATE            Path to activate script (e.g., /path/to/venv/bin/activate)
#   CONDA_ENV                Conda env name to use (runs via 'conda run')
#   REPO_ROOT                Override repo root (default: parent of this script)
#   PYTHON_EXEC              Python executable to use (default: python3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUN_FILE="${REPO_ROOT}/experiments/ph_combo_sweep.py"

PYTHON_CMD=()
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
  PYTHON_CMD=(python)
elif [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_EXE=""
  if [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
  elif [ -x "$HOME/anaconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/anaconda3/bin/conda"
  elif command -v module >/dev/null 2>&1; then
    module load Anaconda3/2024.02 || true
    CONDA_EXE="$(command -v conda || true)"
  fi

  if [ -z "$CONDA_EXE" ]; then
    echo "[ERROR] conda not found. Set CONDA_ENV, VENV_ACTIVATE, or PYTHON_EXEC explicitly."
    exit 1
  fi

  PYTHON_CMD=("$CONDA_EXE" run -n "$CONDA_ENV" python)
elif [[ -n "${PYTHON_EXEC:-}" ]]; then
  PYTHON_CMD=("${PYTHON_EXEC}")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  PYTHON_CMD=(python)
fi

declare -a TIME_STR_ARG=()
if [[ -n "${TIME_STR:-}" ]]; then
  TIME_STR_ARG=("--time-str" "${TIME_STR}")
fi

MAX_WORKERS_VALUE="${MAX_WORKERS:-${NSLOTS:-1}}"
declare -a MAX_WORKERS_ARG=("--max-workers" "${MAX_WORKERS_VALUE}")

GTPB_VALUE="${GUROBI_THREADS_PER_BUNDLE:-}"
if [[ -z "${GTPB_VALUE}" ]]; then
  if [[ "${MAX_WORKERS_VALUE}" -gt 0 ]]; then
    GTPB_VALUE=$(( ${NSLOTS:-1} / ${MAX_WORKERS_VALUE} ))
    if [[ "${GTPB_VALUE}" -lt 1 ]]; then
      GTPB_VALUE=1
    fi
  else
    GTPB_VALUE=1
  fi
fi
declare -a GTPB_ARG=("--gurobi-threads-per-bundle" "${GTPB_VALUE}")

declare -a COMBO_ARGS=()
if [[ -n "${COMBOS:-}" ]]; then
  for combo in ${COMBOS}; do
    COMBO_ARGS+=("--combo" "${combo}")
  done
fi

declare -a RESULTS_ROOT_ARG=()
if [[ -n "${RESULTS_ROOT:-}" ]]; then
  RESULTS_ROOT_ARG=("--results-root" "${RESULTS_ROOT}")
fi

declare -a PH_WORK_ROOT_ARG=()
if [[ -n "${PH_WORK_ROOT:-}" ]]; then
  PH_WORK_ROOT_ARG=("--ph-work-root" "${PH_WORK_ROOT}")
fi

declare -a OUT_ARG=()
if [[ -n "${OUT:-}" ]]; then
  OUT_ARG=("--out" "${OUT}")
fi

declare -a EXTRA_ARGS_LIST=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_LIST=(${EXTRA_ARGS})
fi

env PYTHONPATH="${PYTHONPATH}" "${PYTHON_CMD[@]}" "${RUN_FILE}" \
  "${TIME_STR_ARG[@]:-}" \
  "${MAX_WORKERS_ARG[@]:-}" \
  "${GTPB_ARG[@]:-}" \
  "${COMBO_ARGS[@]:-}" \
  "${RESULTS_ROOT_ARG[@]:-}" \
  "${PH_WORK_ROOT_ARG[@]:-}" \
  "${OUT_ARG[@]:-}" \
  "${EXTRA_ARGS_LIST[@]:-}"
