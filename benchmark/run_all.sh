#!/bin/bash
set -euo pipefail

ROOT_DIR="$(pwd)"
if [[ ! -d "$ROOT_DIR/benchmark" ]]; then
  echo "Run benchmark/run_all.sh from the repository root." >&2
  exit 1
fi
cd "$ROOT_DIR"

MODE="${1:-slurm}"

if [[ "$MODE" == "slurm" ]]; then
  bash benchmark/launch_all_slurm.sh
  exit 0
fi

if [[ "$MODE" == "local" ]]; then
  python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/01_prepare_datasets.ipynb
  python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/02_density_by_length.ipynb
  python benchmark/run_benchmark.py --mode all
  python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/04_evaluate_vdjdb.ipynb
  python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/05_evaluate_yfv_enrichment.ipynb
  python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/06_evaluate_yfv_known_clonotypes.ipynb
  python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/07_evaluate_yfv_cross_donor_overlap.ipynb
  python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark/notebooks/08_summary_and_method_selection.ipynb
  exit 0
fi

echo "Usage: bash benchmark/run_all.sh [slurm|local]" >&2
exit 1
