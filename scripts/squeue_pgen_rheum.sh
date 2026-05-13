#!/bin/sh

#SBATCH --job-name=olga_rheum_redcea
#SBATCH --cpus-per-task=48
#SBATCH --mem=232gb
#SBATCH --time=8:00:00
#SBATCH --output=olga_rheum_redcea.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=medium

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
TARGET_GLOB="/projects/immunestatus/rheum/redcea/runs/as_*/*_enriched_clonotypes_tcremp.tsv"

echo "Script dir: $SCRIPT_DIR"
echo "Searching files: $TARGET_GLOB"

found_any=0
for f in $TARGET_GLOB; do
    if [ ! -f "$f" ]; then
        continue
    fi
    found_any=1
    echo "=== Processing $f ==="
    python "$SCRIPT_DIR/pgens_compute.py" --file --name "$f" --processes "${SLURM_CPUS_PER_TASK:-48}"
done

if [ "$found_any" -eq 0 ]; then
    echo "No files found for pattern: $TARGET_GLOB" >&2
    exit 1
fi
