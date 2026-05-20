#!/bin/bash
#SBATCH --job-name=redcea_bench_07
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark/slurm/logs/%x.%j.out

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${BENCHMARK_ENV_ACTIVATE:-}" ]]; then
  source "$BENCHMARK_ENV_ACTIVATE"
fi

python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/07_evaluate_yfv_cross_donor_overlap.ipynb
