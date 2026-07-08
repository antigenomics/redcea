#!/bin/bash
# Usage:
#   sbatch scripts/yfv/run_redcea_one_yfv_sample.sbatch P1_F1

#SBATCH --job-name=redcea_yfv_one
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=08:00:00
#SBATCH --output=logs_yfv_redcea/%x_%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=medium

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <DONOR_ID>" >&2
  echo "Example: sbatch $0 P1_F1" >&2
  exit 1
fi

DONOR_ID="$1"
AIRR_DIR="/projects/immunestatus/pogorelyy/airr_format"
OUT_ROOT="/projects/immunestatus/pogorelyy/redcea/yfv_single_sample"
LOG_DIR="logs_yfv_redcea"

CHAIN="TRB"
NPROC="${SLURM_CPUS_PER_TASK:-16}"
CLUSTER_ALGO="vdbscan"
K_NEIGHBORS="20"
EPS_K_NEIGHBORS="8"
CLUSTER_MIN_SAMPLES="5"

SUBJECT="${DONOR_ID%%_*}"
REPLICATE="${DONOR_ID#*_}"
PRE_FILE="${AIRR_DIR}/${SUBJECT}_0_${REPLICATE}_with_1.txt"
POST_FILE="${AIRR_DIR}/${SUBJECT}_15_${REPLICATE}.txt"
RUN_DIR="${OUT_ROOT}/${DONOR_ID}"
PREFIX="yfv_${DONOR_ID}_counts"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [[ ! -f "$PRE_FILE" ]]; then
  echo "Background AIRR file not found: $PRE_FILE" >&2
  exit 1
fi

if [[ ! -f "$POST_FILE" ]]; then
  echo "Sample AIRR file not found: $POST_FILE" >&2
  exit 1
fi

echo "Running RedCEA for ${DONOR_ID}"
echo "Sample:      ${POST_FILE}"
echo "Background:  ${PRE_FILE}"
echo "Output dir:  ${RUN_DIR}"
echo "Prefix:      ${PREFIX}"

redcea \
  --sample "$POST_FILE" \
  --background "$PRE_FILE" \
  --output "$RUN_DIR" \
  --prefix "$PREFIX" \
  --chain "$CHAIN" \
  --nproc "$NPROC" \
  --cluster-algo "$CLUSTER_ALGO" \
  --k-neighbors "$K_NEIGHBORS" \
  --eps-k-neighbors "$EPS_K_NEIGHBORS" \
  --cluster-min-samples "$CLUSTER_MIN_SAMPLES" \
  --enrichment-test zbinom \
  --use-clonotype-counts
