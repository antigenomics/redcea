#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

AIRR_DIR="${AIRR_DIR:-/projects/immunestatus/rheum/airr_format}"
EMB_DIR="${EMB_DIR:-/projects/immunestatus/rheum/tcremp}"
GRID_NAME="${GRID_NAME:-rheum_vdbscan_leiden_grid}"
OUT_DIR="${OUT_DIR:-/projects/immunestatus/rheum/redcea/${GRID_NAME}}"
RUNS_DIR="${OUT_DIR}/runs"
LOG_DIR="${OUT_DIR}/logs"
MANIFEST_PATH="${OUT_DIR}/manifest.tsv"

BACKGROUND_FILE="${BACKGROUND_FILE:-${AIRR_DIR}/joint_hd_b27pos.tsv}"
BACKGROUND_EMB="${BACKGROUND_EMB:-${EMB_DIR}/joint_hd_b27pos_embeddings.parquet}"

JOB_PREFIX="${JOB_PREFIX:-rheum_vdbscan_leiden}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHAIN="${CHAIN:-TRB}"
NPROC="${NPROC:-16}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-32gb}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
PARTITION="${PARTITION:-medium}"
CONSTRAINT="${CONSTRAINT:-hpc}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-64}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_ON_MISSING_SAMPLE="${FAIL_ON_MISSING_SAMPLE:-1}"
FAIL_ON_MISSING_SAMPLE_EMBEDDING="${FAIL_ON_MISSING_SAMPLE_EMBEDDING:-1}"
ENABLE_DEBUG_SAVE_INTERMEDIATE="${ENABLE_DEBUG_SAVE_INTERMEDIATE:-0}"
ENABLE_AUX_METRICS="${ENABLE_AUX_METRICS:-0}"
N_BG_POINTS="${N_BG_POINTS:-}"

# A moderate default grid for server-side sweeps.
# Override any of these via env, e.g.:
#   K_NEIGHBORS_VALUES=12,16,20 LEIDEN_RESOLUTION_VALUES=0.2,0.5,1.0 \
#   bash scripts/rheum/launch_vdbscan_leiden_grid.sh
CORE_MIN_SAMPLES_VALUES="${CORE_MIN_SAMPLES_VALUES:-3}"
K_NEIGHBORS_VALUES="${K_NEIGHBORS_VALUES:-4,8,12}"
EPS_K_NEIGHBORS_VALUES="${EPS_K_NEIGHBORS_VALUES:-4,8,12}"
EPS_ESTIMATION_BASED_ON_VALUES="${EPS_ESTIMATION_BASED_ON_VALUES:-sample,background}"
VDBSCAN_SYM_RULE_VALUES="${VDBSCAN_SYM_RULE_VALUES:-asymmetric}"
LEIDEN_RESOLUTION_VALUES="${LEIDEN_RESOLUTION_VALUES:-0.5,1.0,1.5}"
CLUSTER_PC_COMPONENTS_VALUES="${CLUSTER_PC_COMPONENTS_VALUES:-50}"
ENRICHMENT_TEST_VALUES="${ENRICHMENT_TEST_VALUES:-zbinom}"

RHEUM_SAMPLES_CSV="${RHEUM_SAMPLES_CSV:-as_Abd_PB_F,as_Abr_PB_F,as_Ash-110_PB_F_p0,as_Ash-111_PB_F_p0,as_Bal_PB_F,as_Bel_PB_F,as_Bost_PB_F,as_Chaad_PB_F,as_Dv_PB_F,as_Evst_PB_F,as_Gar_PB_F,as_GE_PB_F,as_Gonch_PB_F,as_i70_PB_F,as_Kal_PB_F,as_Kud_PB_F,as_Luk_PB_F,as_Mart_PB_F,as_Mikh_PB_F,as_Shep_PB_F,as_Tsib_PB_F,as_TwHM1_PB_F,as_Uv_PB_F,as_Vas_PB_F,as_Vats_PB_F,as_Vol_PB_F,as_Zakh_PB_F}"

mkdir -p "${OUT_DIR}" "${RUNS_DIR}" "${LOG_DIR}"

if [[ ! -f "${BACKGROUND_FILE}" ]]; then
  echo "Background AIRR file not found: ${BACKGROUND_FILE}" >&2
  exit 1
fi

if [[ ! -f "${BACKGROUND_EMB}" ]]; then
  echo "Background embedding not found: ${BACKGROUND_EMB}" >&2
  echo "Run scripts/rheum/sbatch_joint_hd_background.sh first." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find PYTHON_BIN=${PYTHON_BIN} in PATH." >&2
  exit 1
fi

export AIRR_DIR
export EMB_DIR
export RUNS_DIR
export MANIFEST_PATH
export FAIL_ON_MISSING_SAMPLE
export FAIL_ON_MISSING_SAMPLE_EMBEDDING
export ENABLE_AUX_METRICS
export CORE_MIN_SAMPLES_VALUES
export K_NEIGHBORS_VALUES
export EPS_K_NEIGHBORS_VALUES
export EPS_ESTIMATION_BASED_ON_VALUES
export VDBSCAN_SYM_RULE_VALUES
export LEIDEN_RESOLUTION_VALUES
export CLUSTER_PC_COMPONENTS_VALUES
export ENRICHMENT_TEST_VALUES
export RHEUM_SAMPLES_CSV

eval "$(
"${PYTHON_BIN}" - <<'PY'
import csv
import itertools
import os
import sys
from pathlib import Path


def parse_list(name: str, cast):
    raw = os.environ.get(name, "").strip()
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(cast(token))
    if not values:
        raise SystemExit(f"{name} must contain at least one value")
    return values


def normalize_tag_value(value):
    text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_")


airr_dir = Path(os.environ["AIRR_DIR"])
emb_dir = Path(os.environ["EMB_DIR"])
runs_dir = Path(os.environ["RUNS_DIR"])
manifest_path = Path(os.environ["MANIFEST_PATH"])
fail_on_missing_sample = os.environ.get("FAIL_ON_MISSING_SAMPLE", "1") == "1"
fail_on_missing_sample_embedding = os.environ.get("FAIL_ON_MISSING_SAMPLE_EMBEDDING", "1") == "1"
enable_aux_metrics = os.environ.get("ENABLE_AUX_METRICS", "0") == "1"

samples = parse_list("RHEUM_SAMPLES_CSV", str)
core_min_samples_values = parse_list("CORE_MIN_SAMPLES_VALUES", int)
k_neighbors_values = parse_list("K_NEIGHBORS_VALUES", int)
eps_k_neighbors_values = parse_list("EPS_K_NEIGHBORS_VALUES", int)
eps_estimation_based_on_values = parse_list("EPS_ESTIMATION_BASED_ON_VALUES", str)
vdbscan_sym_rule_values = parse_list("VDBSCAN_SYM_RULE_VALUES", str)
leiden_resolution_values = parse_list("LEIDEN_RESOLUTION_VALUES", float)
cluster_pc_components_values = parse_list("CLUSTER_PC_COMPONENTS_VALUES", int)
enrichment_test_values = parse_list("ENRICHMENT_TEST_VALUES", str)

if enable_aux_metrics and any(test != "zbinom" for test in enrichment_test_values):
    raise SystemExit("ENABLE_AUX_METRICS=1 requires ENRICHMENT_TEST_VALUES to contain only zbinom")

configs = []
for (
    core_min_samples,
    k_neighbors,
    eps_k_neighbors,
    eps_estimation_based_on,
    vdbscan_sym_rule,
    leiden_resolution,
    cluster_pc_components,
    enrichment_test,
) in itertools.product(
    core_min_samples_values,
    k_neighbors_values,
    eps_k_neighbors_values,
    eps_estimation_based_on_values,
    vdbscan_sym_rule_values,
    leiden_resolution_values,
    cluster_pc_components_values,
    enrichment_test_values,
):
    if eps_k_neighbors > k_neighbors:
        continue
    tag = "__".join(
        [
            "algo_vdbscan_leiden",
            f"kn_{normalize_tag_value(k_neighbors)}",
            f"ekn_{normalize_tag_value(eps_k_neighbors)}",
            f"ms_{normalize_tag_value(core_min_samples)}",
            f"eps_{normalize_tag_value(eps_estimation_based_on)}",
            f"sym_{normalize_tag_value(vdbscan_sym_rule)}",
            f"lr_{normalize_tag_value(leiden_resolution)}",
            f"pc_{normalize_tag_value(cluster_pc_components)}",
            f"test_{normalize_tag_value(enrichment_test)}",
        ]
    )
    configs.append(
        {
            "core_min_samples": core_min_samples,
            "k_neighbors": k_neighbors,
            "eps_k_neighbors": eps_k_neighbors,
            "eps_estimation_based_on": eps_estimation_based_on,
            "vdbscan_sym_rule": vdbscan_sym_rule,
            "leiden_resolution": leiden_resolution,
            "cluster_pc_components": cluster_pc_components,
            "enrichment_test": enrichment_test,
            "tag": tag,
        }
    )

rows = []
missing_samples = []
missing_sample_embeddings = []
active_samples = 0

for sample_name in samples:
    sample_file = airr_dir / f"{sample_name}.tsv"
    if not sample_file.is_file():
        missing_samples.append(str(sample_file))
        continue

    active_samples += 1
    sample_embedding = emb_dir / f"{sample_name}_embeddings.parquet"
    if not sample_embedding.is_file():
        missing_sample_embeddings.append(str(sample_embedding))
    sample_embedding_value = str(sample_embedding)

    for config in configs:
        out_dir = runs_dir / sample_name / config["tag"]
        prefix = f"{sample_name}__{config['tag']}"
        rows.append(
            {
                "sample_name": sample_name,
                "sample_file": str(sample_file),
                "sample_embedding": sample_embedding_value,
                "core_min_samples": str(config["core_min_samples"]),
                "k_neighbors": str(config["k_neighbors"]),
                "eps_k_neighbors": str(config["eps_k_neighbors"]),
                "eps_estimation_based_on": str(config["eps_estimation_based_on"]),
                "vdbscan_sym_rule": str(config["vdbscan_sym_rule"]),
                "leiden_resolution": str(config["leiden_resolution"]),
                "cluster_pc_components": str(config["cluster_pc_components"]),
                "enrichment_test": str(config["enrichment_test"]),
                "out_dir": str(out_dir),
                "prefix": prefix,
            }
        )

if missing_samples and fail_on_missing_sample:
    for path in missing_samples:
        print(f"Missing sample AIRR file: {path}", file=sys.stderr)
    raise SystemExit("Aborting because FAIL_ON_MISSING_SAMPLE=1")

if missing_sample_embeddings and fail_on_missing_sample_embedding:
    for path in missing_sample_embeddings:
        print(f"Missing sample embedding: {path}", file=sys.stderr)
    raise SystemExit("Aborting because FAIL_ON_MISSING_SAMPLE_EMBEDDING=1")

if not rows:
    raise SystemExit("No run rows were generated")

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "sample_name",
            "sample_file",
            "sample_embedding",
            "core_min_samples",
            "k_neighbors",
            "eps_k_neighbors",
            "eps_estimation_based_on",
            "vdbscan_sym_rule",
            "leiden_resolution",
            "cluster_pc_components",
            "enrichment_test",
            "out_dir",
            "prefix",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)

if missing_samples and not fail_on_missing_sample:
    for path in missing_samples:
        print(f"Warning: skipping missing sample AIRR file: {path}", file=sys.stderr)

if missing_sample_embeddings and not fail_on_missing_sample_embedding:
    for path in missing_sample_embeddings:
        print(f"Warning: missing sample embedding will be computed on demand: {path}", file=sys.stderr)

print(f"export CONFIG_COUNT={len(configs)}")
print(f"export SAMPLE_COUNT={active_samples}")
print(f"export MANIFEST_ROWS={len(rows)}")
print(f"export MISSING_SAMPLE_COUNT={len(missing_samples)}")
print(f"export MISSING_SAMPLE_EMBEDDING_COUNT={len(missing_sample_embeddings)}")
PY
)"

echo "Prepared rheum vdbscan_leiden grid manifest:"
echo "  samples:         ${SAMPLE_COUNT}"
echo "  param configs:   ${CONFIG_COUNT}"
echo "  total run rows:  ${MANIFEST_ROWS}"
echo "  manifest:        ${MANIFEST_PATH}"
if [[ "${MISSING_SAMPLE_COUNT}" != "0" ]]; then
  echo "  missing samples: ${MISSING_SAMPLE_COUNT}"
fi
if [[ "${MISSING_SAMPLE_EMBEDDING_COUNT}" != "0" ]]; then
  echo "  missing sample embeddings: ${MISSING_SAMPLE_EMBEDDING_COUNT}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, not submitting Slurm jobs."
  exit 0
fi

if [[ "${MANIFEST_ROWS}" -le 0 ]]; then
  echo "Manifest is empty; nothing to submit." >&2
  exit 1
fi

ARRAY_SPEC="0-$((MANIFEST_ROWS - 1))"
if [[ -n "${ARRAY_CONCURRENCY}" && "${ARRAY_CONCURRENCY}" != "0" ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_CONCURRENCY}"
fi

export BACKGROUND_FILE
export BACKGROUND_EMB
export PYTHON_BIN
export CHAIN
export NPROC
export FAIL_ON_MISSING_SAMPLE_EMBEDDING
export ENABLE_DEBUG_SAVE_INTERMEDIATE
export ENABLE_AUX_METRICS
export N_BG_POINTS
export LOG_DIR

JOB_ID="$(
  sbatch --parsable \
    --job-name="${JOB_PREFIX}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --constraint="${CONSTRAINT}" \
    --partition="${PARTITION}" \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/%x_%A_%a.log" \
    --export=ALL <<'EOF'
#!/bin/bash
set -euo pipefail

STATUS_LOG="${LOG_DIR}/grid_status.log"
LOCKFILE="${STATUS_LOG}.lock"
mkdir -p "$(dirname "${STATUS_LOG}")"
touch "${STATUS_LOG}"

log_status() {
  local msg="$1"
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}"
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

on_exit() {
  local exit_code=$?
  local end_ts
  local dur
  end_ts="$(date +%s)"
  dur=$(( end_ts - START_TS_EPOCH ))
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "END_OK   job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} dur_s=${dur}"
  else
    log_status "END_FAIL job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} dur_s=${dur} exit_code=${exit_code}"
  fi
}
trap on_exit EXIT

LINE_NO=$((SLURM_ARRAY_TASK_ID + 2))
ROW="$(sed -n "${LINE_NO}p" "${MANIFEST_PATH}")"
if [[ -z "${ROW}" ]]; then
  echo "No manifest row found for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST_PATH}" >&2
  exit 1
fi

IFS=$'\t' read -r SAMPLE_NAME SAMPLE_FILE SAMPLE_EMB CORE_MIN_SAMPLES K_NEIGHBORS EPS_K_NEIGHBORS EPS_ESTIMATION_BASED_ON VDBSCAN_SYM_RULE LEIDEN_RESOLUTION CLUSTER_PC_COMPONENTS ENRICHMENT_TEST OUT_DIR PREFIX <<< "${ROW}"

if [[ ! -f "${SAMPLE_FILE}" ]]; then
  echo "Sample AIRR file not found: ${SAMPLE_FILE}" >&2
  exit 1
fi

if [[ ! -f "${BACKGROUND_FILE}" ]]; then
  echo "Background AIRR file not found at runtime: ${BACKGROUND_FILE}" >&2
  exit 1
fi

if [[ ! -f "${BACKGROUND_EMB}" ]]; then
  echo "Background embedding not found at runtime: ${BACKGROUND_EMB}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  -m
  redcea.redcea
  -is "${SAMPLE_FILE}"
  -ib "${BACKGROUND_FILE}"
  -o "${OUT_DIR}"
  -e "${PREFIX}"
  -c "${CHAIN}"
  -np "${NPROC}"
  --cluster-algo "vdbscan_leiden"
  -cms "${CORE_MIN_SAMPLES}"
  -kn "${K_NEIGHBORS}"
  -ekn "${EPS_K_NEIGHBORS}"
  --eps-estimation-based-on "${EPS_ESTIMATION_BASED_ON}"
  --vdbscan-sym-rule "${VDBSCAN_SYM_RULE}"
  --leiden-resolution "${LEIDEN_RESOLUTION}"
  -npc "${CLUSTER_PC_COMPONENTS}"
  --enrichment-test "${ENRICHMENT_TEST}"
  -be "${BACKGROUND_EMB}"
)

if [[ -n "${SAMPLE_EMB}" ]]; then
  if [[ ! -f "${SAMPLE_EMB}" && "${FAIL_ON_MISSING_SAMPLE_EMBEDDING}" == "1" ]]; then
    echo "Sample embedding not found at runtime: ${SAMPLE_EMB}" >&2
    exit 1
  fi
  CMD+=(-se "${SAMPLE_EMB}")
fi

if [[ -n "${N_BG_POINTS}" ]]; then
  CMD+=(--n-bg-points "${N_BG_POINTS}")
fi

if [[ "${ENABLE_DEBUG_SAVE_INTERMEDIATE}" == "1" ]]; then
  CMD+=(--debug-save-intermediate --debug-output-dir "${OUT_DIR}/debug")
fi

if [[ "${ENABLE_AUX_METRICS}" == "1" ]]; then
  CMD+=(--add-auxiliary-cluster-metrics)
fi

log_status "START    job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} sample=${SAMPLE_NAME} ms=${CORE_MIN_SAMPLES} kn=${K_NEIGHBORS} ekn=${EPS_K_NEIGHBORS} eps=${EPS_ESTIMATION_BASED_ON} sym=${VDBSCAN_SYM_RULE} lr=${LEIDEN_RESOLUTION} pc=${CLUSTER_PC_COMPONENTS} out=${OUT_DIR}"

echo "=== RUN CONFIG ==="
echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SAMPLE=${SAMPLE_NAME}"
echo "OUT_DIR=${OUT_DIR}"
echo "PREFIX=${PREFIX}"
echo "MS=${CORE_MIN_SAMPLES}  KN=${K_NEIGHBORS}  EKN=${EPS_K_NEIGHBORS}"
echo "EPS=${EPS_ESTIMATION_BASED_ON}  VSYM=${VDBSCAN_SYM_RULE}  LR=${LEIDEN_RESOLUTION}"
echo "PC=${CLUSTER_PC_COMPONENTS}  TEST=${ENRICHMENT_TEST}"
echo "==================="

"${CMD[@]}"
EOF
)"

echo "Submitted Slurm array job:"
echo "  job id:    ${JOB_ID}"
echo "  array:     ${ARRAY_SPEC}"
echo "  logs dir:  ${LOG_DIR}"
echo "  status log:${LOG_DIR}/grid_status.log"
