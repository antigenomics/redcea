#!/bin/bash
#SBATCH --job-name=redcea_bench_08
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark/slurm/logs/%x.%j.out

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"
if [[ ! -f benchmark/notebooks/08_summary_and_method_selection.ipynb ]]; then
  echo "Expected to run from repository root; benchmark notebook path was not found." >&2
  exit 1
fi

if [[ -n "${BENCHMARK_ENV_ACTIVATE:-}" ]]; then
  source "$BENCHMARK_ENV_ACTIVATE"
fi

echo "[$(date)] Starting benchmark summary notebook in $(pwd)"
python -u -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/08_summary_and_method_selection.ipynb
echo "[$(date)] Finished benchmark summary notebook"
