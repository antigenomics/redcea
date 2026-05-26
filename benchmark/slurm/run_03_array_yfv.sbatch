#!/bin/bash
#SBATCH --job-name=redcea_bench_03_yfv
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --partition=medium
#SBATCH --output=benchmark/slurm/logs/%x.%A_%a.out

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"
if [[ ! -f benchmark/run_benchmark.py ]]; then
  echo "Expected to run from repository root; benchmark/run_benchmark.py was not found." >&2
  exit 1
fi

if [[ -n "${BENCHMARK_ENV_ACTIVATE:-}" ]]; then
  source "$BENCHMARK_ENV_ACTIVATE"
fi

PYTHON_BIN="${BENCHMARK_PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

echo "[$(date)] Starting YFV array task ${SLURM_ARRAY_TASK_ID} in $(pwd)"
"$PYTHON_BIN" -u benchmark/run_benchmark.py --mode single --single-manifest-path results/run_metadata/clustering_manifest_yfv.tsv --single-row-index "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Finished YFV array task ${SLURM_ARRAY_TASK_ID}"
