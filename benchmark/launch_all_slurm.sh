#!/bin/bash
set -euo pipefail

ROOT_DIR="$(pwd)"
if [[ ! -d "$ROOT_DIR/benchmark/slurm" ]]; then
  echo "Run benchmark/launch_all_slurm.sh from the repository root." >&2
  exit 1
fi
SLURM_DIR="$ROOT_DIR/benchmark/slurm"

submit_job() {
  local dependency="$1"
  local script_path="$2"
  local output
  if [[ -n "$dependency" ]]; then
    output="$(sbatch --parsable --dependency=afterok:${dependency} "$script_path")"
  else
    output="$(sbatch --parsable "$script_path")"
  fi
  echo "$output"
}

job01="$(submit_job "" "$SLURM_DIR/run_01_prepare_datasets.sbatch")"
job02="$(submit_job "$job01" "$SLURM_DIR/run_02_density_by_length.sbatch")"
job03="$(submit_job "$job01" "$SLURM_DIR/run_03_build_manifests.sbatch")"
job03_dispatch="$(sbatch --parsable --dependency=afterok:${job03} --export=ALL,DENSITY_JOB_ID=${job02} "$SLURM_DIR/run_03_dispatch_arrays.sbatch")"

echo "Submitted benchmark jobs:"
echo "  01_prepare_datasets:            $job01"
echo "  02_density_by_length:           $job02"
echo "  03_build_manifests:             $job03"
echo "  03_dispatch_arrays:             $job03_dispatch"
