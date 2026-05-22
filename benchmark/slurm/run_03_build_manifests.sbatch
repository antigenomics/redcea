#!/bin/bash
#SBATCH --job-name=redcea_bench_03_manifest
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

python benchmark/run_benchmark.py --mode manifest --grid-size "${BENCHMARK_GRID_SIZE:-small}"
