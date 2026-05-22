#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_benchmark.sh small
  bash scripts/run_benchmark.sh large

The script:
  1. cleans previous benchmark outputs
  2. submits the Slurm benchmark pipeline
  3. uses the requested hyperparameter grid size

Any other arguments are forwarded to scripts/run_redcea_compare_benchmark.py
for backward compatibility.
EOF
}

submit_benchmark() {
  local grid_size="$1"
  if [[ "$grid_size" != "small" && "$grid_size" != "large" ]]; then
    echo "Unsupported grid size: $grid_size" >&2
    usage >&2
    exit 1
  fi

  rm -rf results/redcea_runs
  rm -f results/clustering_assignments/*.parquet
  rm -f results/run_metadata/clustering_run_parts/*.tsv
  rm -f results/run_metadata/clustering_runs.tsv
  rm -f results/run_metadata/clustering_grid_manifest.tsv
  rm -f results/run_metadata/clustering_manifest_vdjdb.tsv
  rm -f results/run_metadata/clustering_manifest_yfv.tsv
  rm -f results/run_metadata/clustering_execution_manifest.tsv
  rm -f results/enrichment/*.tsv
  rm -f results/metrics/*.tsv
  rm -f results/density/*.tsv
  rm -f reports/clustering_strategy_summary.md
  rm -rf figures/clustering_strategy
  rm -f data/processed/benchmark_dataset_manifest.tsv
  rm -f data/processed/known_yfv_vdjdb_clonotypes.tsv

  jid01=$(sbatch --parsable benchmark/slurm/run_01_prepare_datasets.sbatch)
  jid02=$(sbatch --parsable --dependency=afterok:$jid01 benchmark/slurm/run_02_density_by_length.sbatch)
  jid03m=$(sbatch --parsable --dependency=afterok:$jid01 --export=ALL,BENCHMARK_GRID_SIZE="$grid_size" benchmark/slurm/run_03_build_manifests.sbatch)

  while [[ ! -f results/run_metadata/clustering_manifest_vdjdb.tsv || ! -f results/run_metadata/clustering_manifest_yfv.tsv ]]; do
    sleep 10
  done

  vdjdb_count=$(python - <<'PY'
import pandas as pd
print(len(pd.read_csv("results/run_metadata/clustering_manifest_vdjdb.tsv", sep="\t")))
PY
)

  yfv_count=$(python - <<'PY'
import pandas as pd
print(len(pd.read_csv("results/run_metadata/clustering_manifest_yfv.tsv", sep="\t")))
PY
)

  jid03v=$(sbatch --parsable --dependency=afterok:$jid03m --array=1-${vdjdb_count} benchmark/slurm/run_03_array_vdjdb.sbatch)
  jid03y=$(sbatch --parsable --dependency=afterok:$jid03m --array=1-${yfv_count} benchmark/slurm/run_03_array_yfv.sbatch)
  jid03c=$(sbatch --parsable --dependency=afterok:$jid03v:$jid03y benchmark/slurm/run_03_consolidate_metadata.sbatch)
  jid04=$(sbatch --parsable --dependency=afterok:$jid03c benchmark/slurm/run_04_evaluate_vdjdb.sbatch)
  jid05=$(sbatch --parsable --dependency=afterok:$jid03c benchmark/slurm/run_05_evaluate_yfv_enrichment.sbatch)
  jid06=$(sbatch --parsable --dependency=afterok:$jid05 benchmark/slurm/run_06_evaluate_yfv_known_clonotypes.sbatch)
  jid07=$(sbatch --parsable --dependency=afterok:$jid05 benchmark/slurm/run_07_evaluate_yfv_cross_donor_overlap.sbatch)
  jid08=$(sbatch --parsable --dependency=afterok:$jid02:$jid04:$jid05:$jid06:$jid07 benchmark/slurm/run_08_summary_and_method_selection.sbatch)

  cat <<EOF
Submitted benchmark grid_size=$grid_size
  prepare:      $jid01
  density:      $jid02
  manifests:    $jid03m
  vdjdb array:  $jid03v
  yfv array:    $jid03y
  consolidate:  $jid03c
  eval vdjdb:   $jid04
  eval enrich:  $jid05
  eval known:   $jid06
  eval overlap: $jid07
  summary:      $jid08
EOF
}

if [[ $# -ge 1 && ( "$1" == "small" || "$1" == "large" ) ]]; then
  submit_benchmark "$1"
  exit 0
fi

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

python scripts/run_redcea_compare_benchmark.py "$@"
