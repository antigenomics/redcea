#!/bin/bash
#SBATCH --job-name=tcrempnet_YLQ_vdbscan_grid
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=2:00:00
#SBATCH --constraint=hpc
#SBATCH --partition=short
#SBATCH --array=0-23
#SBATCH --output=/projects/immunestatus/vdjdb/tcrempnet_logs/%x_%A_%a.log

set -euo pipefail

# =========================
# Global status log (single file for whole grid)
# =========================
STATUS_LOG="/projects/immunestatus/vdjdb/tcrempnet_logs/vdbscan_grid_status.log"
LOCKFILE="${STATUS_LOG}.lock"
mkdir -p "$(dirname "${STATUS_LOG}")"
touch "${STATUS_LOG}"

log_status () {
  local MSG="$1"
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] ${MSG}"
  if command -v flock >/dev/null 2>&1; then
    (
      flock -x 200
      echo "${line}" >> "${STATUS_LOG}"
    ) 200>"${LOCKFILE}"
  else
    echo "${line}" >> "${STATUS_LOG}"
  fi
}

START_TS_EPOCH="$(date +%s)"

on_exit () {
  exit_code=$?
  end_ts="$(date +%s)"
  dur=$(( end_ts - START_TS_EPOCH ))
  if [[ $exit_code -eq 0 ]]; then
    log_status "END_OK   job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} dur_s=${dur}"
  else
    log_status "END_FAIL job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} dur_s=${dur} exit_code=${exit_code}"
  fi
}
trap on_exit EXIT

# =========================
# Fixed inputs (your paths)
# =========================
SAMPLE="/projects/immunestatus/vdjdb/airr_format/trb_vdjdb_YLQPRTFLL.tsv"
BACKGROUND="/projects/immunestatus/vdjdb/airr_format/trb_background.tsv"
SAMPLE_EMB="/projects/immunestatus/vdjdb/tcremp/trb_vdjdb_YLQPRTFLL_embeddings.parquet"
BG_EMB="/projects/immunestatus/vdjdb/tcremp/trb_background_embeddings.parquet"

CHAIN="TRB"
NPROC=16
N_BG_POINTS=100000

# Code location (absolute path, avoids wrong cwd issues)
TCREMPNET_PY="/home/evlasova/tcrempnet/tcremp/tcrempnet.py"

# Base output prefix (your old naming style)
BASE_OUT="/projects/immunestatus/vdjdb/tcrempnet_YLQPRTFLL_trb_vdbscan"

# =========================
# Pick vdbscan config by array id
# =========================
eval "$(
python - <<'PY'
import os
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

kns  = [4, 8, 12, 16]
ms   = [3, 5, 10]
sym_rules = ["asymmetric", "min"]

# fixed params
ekn = 4
eps = "all"  # --eps-estimation-based-on

configs = []
for k in kns:
    for m in ms:
        for sr in sym_rules:
            configs.append(dict(kn=k, ms=m, ekn=ekn, eps=eps, vsym=sr))

assert len(configs) == 24, len(configs)
cfg = configs[task_id]

for k, v in cfg.items():
    if isinstance(v, str):
        print(f'export {k.upper()}="{v}"')
    else:
        print(f"export {k.upper()}={v}")
PY
)"

# =========================
# Build output directory name (unique per config)
# =========================
OUT="${BASE_OUT}/kn_${KN}/ekn_${EKN}__ms_${MS}__eps_${EPS}__sym_${VSYM}"
mkdir -p "${OUT}"

# =========================
# Log START
# =========================
log_status "START    job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} algo=vdbscan kn=${KN} ekn=${EKN} ms=${MS} eps=${EPS} vsym=${VSYM} out=${OUT}"

echo "=== RUN CONFIG ==="
echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  ALGO=vdbscan  OUT=${OUT}"
echo "KN=${KN}  EKN=${EKN}  MS=${MS}  EPS=${EPS}  VSYM=${VSYM}"
echo "==================="

# =========================
# Build command (vdbscan only)
# =========================
CMD=(python "${TCREMPNET_PY}"
  --sample "${SAMPLE}"
  --background "${BACKGROUND}"
  --output "${OUT}"
  --chain "${CHAIN}"
  -np "${NPROC}"
  -se "${SAMPLE_EMB}"
  -be "${BG_EMB}"
  --n-bg-points "${N_BG_POINTS}"
  --cluster-algo "vdbscan"
  -kn "${KN}"
  -ms "${MS}"
  -ekn "${EKN}"
  --eps-estimation-based-on "${EPS}"
  --vdbscan-sym-rule "${VSYM}"
)

# =========================
# Run
# =========================
"${CMD[@]}"
