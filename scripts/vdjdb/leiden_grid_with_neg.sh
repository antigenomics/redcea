#!/bin/bash
#SBATCH --job-name=tcrempnet_leiden_grid_trb_top8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=2:00:00
#SBATCH --constraint=hpc
#SBATCH --partition=short
# 5 epitopes * 110 configs = 550 tasks (if you keep the grid below)
#SBATCH --array=0-550
#SBATCH --output=/projects/immunestatus/vdjdb_valid/tcrempnet_logs_trb/%x_%A_%a.log

set -euo pipefail

# =========================
# Global status log (single file for whole grid)
# =========================
STATUS_LOG="/projects/immunestatus/vdjdb_valid/tcrempnet_logs_trb/grid_status.log"
LOCKFILE="${STATUS_LOG}.lock"
mkdir -p "$(dirname "${STATUS_LOG}")"
touch "${STATUS_LOG}"

log_status () {
  local MSG="$1"
  (
    flock -x 200
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${MSG}"
  ) 200>"${LOCKFILE}"
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
# Fixed inputs (paths / knobs)
# =========================
CHAIN="TRB"
NPROC=16
N_BG_POINTS=100000

BACKGROUND="/projects/immunestatus/vdjdb_valid/airr_format/${CHAIN,,}_background.tsv"
BG_EMB="/projects/immunestatus/vdjdb/tcremp/${CHAIN,,}_background_embeddings.parquet"

LOGDIR="/projects/immunestatus/vdjdb_valid/tcrempnet_logs_trb"
mkdir -p "${LOGDIR}"

# =========================
# Epitopes (top list)
# =========================
# NOTE: your screenshot shows 7; add the 8th if needed.
EPITOPES=(
  "GILGFVFTL"
  # "RAKFKQII"
  "GLCTLVAML"
  "NLVPMVATV"
  "YLQPRTFLL"
  # "KRWIILGLNK"
  "YVLDHLIVV"
  # "ADD_8TH_EPITOPE_HERE"
)

# Make epitopes available for python as a single env string
export EPITOPES_STR
EPITOPES_STR="$(IFS=,; echo "${EPITOPES[*]}")"

# =========================
# Pick config by array id
# =========================
eval "$(
python - <<'PY'
import os

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
ep_str = os.environ.get("EPITOPES_STR", "")
epitopes = [x for x in ep_str.split(",") if x.strip()]

# --- Leiden grid (expanded a bit) ---
kns = [2, 4, 8, 16, 25, 40, 64, 80, 100, 150, 200]
lrs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
configs = [(kn, lr) for kn in kns for lr in lrs]

n_ep = len(epitopes)
n_cfg = len(configs)
total = n_ep * n_cfg

if task_id >= total:
    print('export SKIP=1')
    print(f'export TOTAL_TASKS={total}')
    raise SystemExit(0)

ep_i = task_id // n_cfg
cfg_i = task_id % n_cfg

ep = epitopes[ep_i]
kn, lr = configs[cfg_i]

print('export SKIP=0')
print(f'export TOTAL_TASKS={total}')
print(f'export EPITOPE="{ep}"')
print(f'export KN={kn}')
print(f'export LR={lr}')
PY
)"

if [[ "${SKIP}" == "1" ]]; then
  log_status "SKIP     job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} task_id_out_of_range total=${TOTAL_TASKS}"
  echo "SKIP: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL_TASKS=${TOTAL_TASKS}"
  exit 0
fi

# =========================
# Epitope-specific inputs
# =========================
SAMPLE="/projects/immunestatus/vdjdb_valid/airr_format/${CHAIN,,}_vdjdb_${EPITOPE}.tsv"
SAMPLE_EMB="/projects/immunestatus/vdjdb_valid/tcremp/${CHAIN,,}_vdjdb_${EPITOPE}_embeddings.parquet"

# =========================
# Output directory (unique per config)
# =========================
BASE_OUT_PREFIX="/projects/immunestatus/vdjdb_valid/tcrempnet_${EPITOPE}_${CHAIN,,}"
OUT="${BASE_OUT_PREFIX}/leiden/kn_${KN}/lr_${LR}"
mkdir -p "${OUT}"

# =========================
# Log START
# =========================
log_status "START    job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} ep=${EPITOPE} algo=leiden chain=${CHAIN} kn=${KN} lr=${LR} out=${OUT}"

echo "=== RUN CONFIG ==="
echo "TASK_ID=${SLURM_ARRAY_TASK_ID}/${TOTAL_TASKS}  EPITOPE=${EPITOPE}  CHAIN=${CHAIN}  ALGO=leiden"
echo "KN=${KN}  LR=${LR}"
echo "SAMPLE=${SAMPLE}"
echo "SAMPLE_EMB=${SAMPLE_EMB}"
echo "OUT=${OUT}"
echo "==================="

# =========================
# Build & run command (ONLY leiden)
# =========================
CMD=(python /home/evlasova/tcrempnet/tcremp/tcrempnet.py
  --sample "${SAMPLE}"
  --background "${BACKGROUND}"
  --output "${OUT}"
  --chain "${CHAIN}"
  -np "${NPROC}"
  -se "${SAMPLE_EMB}"
  -be "${BG_EMB}"
  --n-bg-points "${N_BG_POINTS}"
  --cluster-algo "leiden"
  -kn "${KN}"
  --leiden-resolution "${LR}"
)

"${CMD[@]}"