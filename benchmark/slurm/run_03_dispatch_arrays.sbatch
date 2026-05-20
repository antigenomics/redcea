#!/bin/bash
#SBATCH --job-name=redcea_bench_03_dispatch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark/slurm/logs/%x.%j.out

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"
if [[ ! -f benchmark/slurm/run_03_array_vdjdb.sbatch ]]; then
  echo "Expected to run from repository root; benchmark/slurm scripts were not found." >&2
  exit 1
fi

if [[ -n "${BENCHMARK_ENV_ACTIVATE:-}" ]]; then
  source "$BENCHMARK_ENV_ACTIVATE"
fi

VDJDB_MANIFEST="results/run_metadata/clustering_manifest_vdjdb.tsv"
YFV_MANIFEST="results/run_metadata/clustering_manifest_yfv.tsv"

count_rows() {
  local file_path="$1"
  python - "$file_path" <<'PY'
from pathlib import Path
import pandas as pd
import sys
path = Path(sys.argv[1])
df = pd.read_csv(path, sep="\t")
print(len(df))
PY
}

VDJDB_COUNT="$(count_rows "$VDJDB_MANIFEST")"
YFV_COUNT="$(count_rows "$YFV_MANIFEST")"
echo "Dispatch working directory: $(pwd)"
echo "VDJdb manifest: ${VDJDB_MANIFEST} rows=${VDJDB_COUNT}"
echo "YFV manifest: ${YFV_MANIFEST} rows=${YFV_COUNT}"

if [[ "$VDJDB_COUNT" -lt 1 ]]; then
  echo "VDJdb manifest is empty: $VDJDB_MANIFEST" >&2
  exit 1
fi

if [[ "$YFV_COUNT" -lt 1 ]]; then
  echo "YFV manifest is empty: $YFV_MANIFEST" >&2
  exit 1
fi

echo "Submitting VDJdb array: --array=1-${VDJDB_COUNT} benchmark/slurm/run_03_array_vdjdb.sbatch"
job_vdjdb="$(sbatch --parsable --array=1-${VDJDB_COUNT} benchmark/slurm/run_03_array_vdjdb.sbatch)"
echo "Submitting YFV array: --array=1-${YFV_COUNT} benchmark/slurm/run_03_array_yfv.sbatch"
job_yfv="$(sbatch --parsable --array=1-${YFV_COUNT} benchmark/slurm/run_03_array_yfv.sbatch)"
# Consolidate should still run if a few array tasks fail; failed runs are recorded in
# per-run metadata and downstream evaluation filters to successful assignments.
echo "Submitting metadata consolidation after arrays finish: ${job_vdjdb}:${job_yfv}"
job03_consolidate="$(sbatch --parsable --dependency=afterany:${job_vdjdb}:${job_yfv} benchmark/slurm/run_03_consolidate_metadata.sbatch)"

echo "Submitting downstream evaluation jobs"
job04="$(sbatch --parsable --dependency=afterok:${job03_consolidate} benchmark/slurm/run_04_evaluate_vdjdb.sbatch)"
job05="$(sbatch --parsable --dependency=afterok:${job03_consolidate} benchmark/slurm/run_05_evaluate_yfv_enrichment.sbatch)"
job06="$(sbatch --parsable --dependency=afterok:${job05} benchmark/slurm/run_06_evaluate_yfv_known_clonotypes.sbatch)"
job07="$(sbatch --parsable --dependency=afterok:${job05} benchmark/slurm/run_07_evaluate_yfv_cross_donor_overlap.sbatch)"
summary_deps="${job04}:${job06}:${job07}"
job08="$(sbatch --parsable --dependency=afterok:${summary_deps} benchmark/slurm/run_08_summary_and_method_selection.sbatch)"

echo "Submitted array jobs:"
echo "  VDJdb array: $job_vdjdb (tasks=${VDJDB_COUNT}, mem=64G)"
echo "  YFV array:   $job_yfv (tasks=${YFV_COUNT}, mem=256G)"
echo "Submitted downstream jobs:"
echo "  03_consolidate_metadata: $job03_consolidate"
echo "  04_evaluate_vdjdb: $job04"
echo "  05_evaluate_yfv_enrichment: $job05"
echo "  06_evaluate_yfv_known_clonotypes: $job06"
echo "  07_evaluate_yfv_cross_donor_overlap: $job07"
echo "  08_summary_and_method_selection: $job08"
if [[ -n "${DENSITY_JOB_ID:-}" ]]; then
  echo "  02_density_by_length runs independently and does not block summary: ${DENSITY_JOB_ID}"
fi
