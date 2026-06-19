#!/bin/bash
#SBATCH --job-name=redcea_bench_01
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark/slurm/logs/%x.%j.out

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"
if [[ ! -f benchmark/prepare_datasets.py ]]; then
  echo "Expected to run from repository root; benchmark/prepare_datasets.py was not found." >&2
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

CMD=("$PYTHON_BIN" benchmark/prepare_datasets.py)
if [[ -n "${BENCHMARK_VDJDB_TARGETS:-}" ]]; then
  CMD+=(--vdjdb-targets "${BENCHMARK_VDJDB_TARGETS}")
fi

"${CMD[@]}"
