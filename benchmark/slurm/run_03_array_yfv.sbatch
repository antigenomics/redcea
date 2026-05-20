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

python benchmark/run_benchmark.py --mode single --single-manifest-path results/run_metadata/clustering_manifest_yfv.tsv --single-row-index "${SLURM_ARRAY_TASK_ID}"
