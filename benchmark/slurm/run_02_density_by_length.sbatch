#!/bin/bash
#SBATCH --job-name=redcea_bench_02
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --partition=medium
#SBATCH --output=benchmark/slurm/logs/%x.%j.out

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"
if [[ ! -f benchmark/notebooks/02_density_by_length.ipynb ]]; then
  echo "Expected to run from repository root; benchmark notebook path was not found." >&2
  exit 1
fi

if [[ -n "${BENCHMARK_ENV_ACTIVATE:-}" ]]; then
  source "$BENCHMARK_ENV_ACTIVATE"
fi

python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/02_density_by_length.ipynb
