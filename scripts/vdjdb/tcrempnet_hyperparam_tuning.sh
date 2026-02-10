#!/bin/bash
#SBATCH --job-name=tcrempnet_YLQ_grid_tra
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=2:00:00
#SBATCH --constraint=hpc
#SBATCH --partition=short
#SBATCH --array=0-147
#SBATCH --output=/projects/immunestatus/vdjdb/tcrempnet_logs_tra/%x_%A_%a.log

set -euo pipefail

# =========================
# Global status log (single file for whole grid)
# =========================
STATUS_LOG="/projects/immunestatus/vdjdb/tcrempnet_logs_tra/grid_status.log"
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
# Fixed inputs (your paths)
# =========================
SAMPLE="/projects/immunestatus/vdjdb/airr_format/tra_vdjdb_YLQPRTFLL.tsv"
BACKGROUND="/projects/immunestatus/vdjdb/airr_format/tra_background.tsv"
SAMPLE_EMB="/projects/immunestatus/vdjdb/tcremp/tra_vdjdb_YLQPRTFLL_embeddings.parquet"
BG_EMB="/projects/immunestatus/vdjdb/tcremp/tra_background_embeddings.parquet"

CHAIN="TRB"
NPROC=16
N_BG_POINTS=100000

# Your old output base name as prefix for all runs
BASE_OUT_PREFIX="/projects/immunestatus/vdjdb/tcrempnet_YLQPRTFLL_tra"

# =========================
# Pick config by array id
# =========================
eval "$(
python - <<'PY'
import os
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

kns  = [4, 8, 12, 16]
res  = [0.5, 1.0, 2.0, 3.0]
sub  = [2.0, 3.0, 4.0]
ms   = [3, 5, 10]
epsm = ["sample", "background", "all"]

configs = []

# leiden: (res, kn) => 4*4=16
for r in res:
    for k in kns:
        configs.append(dict(algo="leiden", lr=r, kn=k))

# hierarchical_leiden: (res, sub, kn) => 4*3*4=48
for r in res:
    for sr in sub:
        for k in kns:
            configs.append(dict(algo="hierarchical_leiden", lr=r, slr=sr, kn=k))

# leiden_dbscan: (res, kn, ms) => 4*4*3=48
for r in res:
    for k in kns:
        for m in ms:
            configs.append(dict(algo="leiden_dbscan", lr=r, kn=k, ms=m))

# vdbscan: (kn, ms, eps_mode) => 4*3*3=36
for k in kns:
    for m in ms:
        for e in epsm:
            configs.append(dict(algo="vdbscan", kn=k, ms=m, eps=e))

assert len(configs) == 148, len(configs)
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
# Maps old "..._trb_vdbscan" style into "..._trb_<algo>"
OUT="${BASE_OUT_PREFIX}_${ALGO}/kn_${KN}"

if [[ "${ALGO}" == "leiden" ]]; then
  OUT="${OUT}/lr_${LR}"
elif [[ "${ALGO}" == "hierarchical_leiden" ]]; then
  OUT="${OUT}/lr_${LR}__slr_${SLR}"
elif [[ "${ALGO}" == "leiden_dbscan" ]]; then
  OUT="${OUT}/lr_${LR}__ms_${MS}"
elif [[ "${ALGO}" == "vdbscan" ]]; then
  OUT="${OUT}/ms_${MS}__eps_${EPS}"
else
  echo "Unknown ALGO=${ALGO}" >&2
  exit 1
fi

mkdir -p "${OUT}"

# =========================
# Log START (after config is known)
# =========================
log_status "START    job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} algo=${ALGO} kn=${KN} lr=${LR:-NA} slr=${SLR:-NA} ms=${MS:-NA} eps=${EPS:-NA} out=${OUT}"

echo "=== RUN CONFIG ==="
echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  ALGO=${ALGO}  KN=${KN}  OUT=${OUT}"
[[ -n "${LR:-}"  ]] && echo "LR=${LR}"
[[ -n "${SLR:-}" ]] && echo "SLR=${SLR}"
[[ -n "${MS:-}"  ]] && echo "MS=${MS}"
[[ -n "${EPS:-}" ]] && echo "EPS=${EPS}"
echo "==================="

# =========================
# Build command
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
  --cluster-algo "${ALGO}"
  -kn "${KN}"
  -c TRA
)

# --- algo-specific flags ---
# IMPORTANT: Adjust flag names if your CLI differs.
# If your script uses -lr (as in your example), you can replace --leiden-resolution with -lr.
if [[ "${ALGO}" == "leiden" ]]; then
  CMD+=(--leiden-resolution "${LR}")
elif [[ "${ALGO}" == "hierarchical_leiden" ]]; then
  CMD+=(--leiden-resolution "${LR}" --leiden-sub-resolution "${SLR}")
elif [[ "${ALGO}" == "leiden_dbscan" ]]; then
  CMD+=(--leiden-resolution "${LR}" -ms "${MS}")
elif [[ "${ALGO}" == "vdbscan" ]]; then
  CMD+=(-ms "${MS}" --eps-estimation-based-on "${EPS}")
fi

# =========================
# Run
# =========================
"${CMD[@]}"
