#!/bin/bash
#SBATCH --job-name=redcea_bench_03_yfv
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=benchmark/slurm/logs/%x.%A_%a.out

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${BENCHMARK_ENV_ACTIVATE:-}" ]]; then
  source "$BENCHMARK_ENV_ACTIVATE"
fi

python benchmark/run_benchmark.py --mode single --single-manifest-path results/run_metadata/clustering_manifest_yfv.tsv --single-row-index "${SLURM_ARRAY_TASK_ID}"
