#!/bin/bash
#SBATCH --job-name=redcea_bench_03_consolidate
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark/slurm/logs/%x.%j.out

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

echo "[$(date)] Starting metadata consolidation in $(pwd)"
python -u benchmark/run_benchmark.py --mode consolidate
echo "[$(date)] Finished metadata consolidation"
