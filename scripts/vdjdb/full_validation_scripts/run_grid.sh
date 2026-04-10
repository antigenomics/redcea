#!/bin/bash
#SBATCH --job-name=tcrempnet_leiden_grid_vdjdb_olga
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=2:00:00
#SBATCH --constraint=hpc
#SBATCH --partition=short
#SBATCH --array=0-550
#SBATCH --output=/projects/immunestatus/vdjdb_olga/tcrempnet_logs/%x_%A_%a.log

set -euo pipefail

# =========================
# Args
# =========================
# usage:
#   sbatch run_grid.sh --chain TRB --tn 10
#   sbatch run_grid.sh --chain TRA --tn 25
CHAIN="TRB"
TN="10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chain) CHAIN="$2"; shift 2;;
    --tn)    TN="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

# =========================
# Global status log
# =========================
LOGROOT="/projects/immunestatus/vdjdb_olga/tcrempnet_logs_${CHAIN,,}_tn${TN}"
STATUS_LOG="${LOGROOT}/grid_status.log"
LOCKFILE="${STATUS_LOG}.lock"
mkdir -p "${LOGROOT}"
touch "${STATUS_LOG}"

log_status () {
  local MSG="$1"
  (
    flock -x 200
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${MSG}"
  ) 200>"${LOCKFILE}"
}

START_TS_EPOCH="$(date +%s)"
trap 'ec=$?; end=$(date +%s); dur=$((end-START_TS_EPOCH)); if [[ $ec -eq 0 ]]; then log_status "END_OK   job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} dur_s=${dur}"; else log_status "END_FAIL job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} dur_s=${dur} exit_code=${ec}"; fi' EXIT

# =========================
# Fixed inputs
# =========================
NPROC=16

AIRR_DIR="/projects/immunestatus/vdjdb_olga/airr_format"
TCREMP_DIR="/projects/immunestatus/vdjdb_olga/tcremp"

BACKGROUND="${AIRR_DIR}/${CHAIN,,}_background_100k.tsv"
BG_EMB="${TCREMP_DIR}/${CHAIN,,}_background_100k_embeddings.parquet"

# =========================
# Epitopes (fixed list)
# =========================
EPITOPES=(
  "GILGFVFTL"
  "GLCTLVAML"
  "NLVPMVATV"
  "YLQPRTFLL"
  "YVLDHLIVV"
)

export EPITOPES_STR
EPITOPES_STR="$(IFS=,; echo "${EPITOPES[*]}")"

# =========================
# Pick config by array id
# =========================
eval "$(
python - <<'PY'
import os

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
epitopes = [x for x in os.environ["EPITOPES_STR"].split(",") if x]

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
  exit 0
fi

# =========================
# Inputs per epitope (tn10/tn25)
# =========================
SAMPLE="${AIRR_DIR}/${CHAIN,,}_vdjdb_${EPITOPE}_tn${TN}.tsv"
SAMPLE_EMB="${TCREMP_DIR}/${CHAIN,,}_vdjdb_${EPITOPE}_tn${TN}_embeddings.parquet"

# =========================
# Output directory
# =========================
BASE_OUT_PREFIX="/projects/immunestatus/vdjdb_olga/tcrempnet_${EPITOPE}_${CHAIN,,}_tn${TN}"
OUT="${BASE_OUT_PREFIX}/leiden/kn_${KN}/lr_${LR}"
mkdir -p "${OUT}"

log_status "START    job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} ep=${EPITOPE} algo=leiden chain=${CHAIN} tn=${TN} kn=${KN} lr=${LR} out=${OUT}"

CMD=(python -m redcea.redcea
  --sample "${SAMPLE}"
  --background "${BACKGROUND}"
  --output "${OUT}"
  --chain "${CHAIN}"
  -np "${NPROC}"
  -se "${SAMPLE_EMB}"
  -be "${BG_EMB}"
  --cluster-algo "leiden"
  -kn "${KN}"
  --leiden-resolution "${LR}"
)

"${CMD[@]}"
