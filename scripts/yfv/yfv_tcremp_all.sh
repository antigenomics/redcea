#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIRR_DIR="/projects/immunestatus/pogorelyy/airr_format"
EMB_DIR="/projects/immunestatus/pogorelyy/tcremp"
LOG_DIR="$SCRIPT_DIR/logs_tcremp"

mkdir -p "$LOG_DIR"

samples=(
  P1
  P2
  Q1
  Q2
  S1
  S2
)

days=(0 15)

for sample in "${samples[@]}"; do
  for day in "${days[@]}"; do
    if [[ "$day" == "0" ]]; then
      sample_file="$AIRR_DIR/${sample}_0_F1_with_1.txt"
      sample_tag="${sample}_0_F1_with_1"
    else
      sample_file="$AIRR_DIR/${sample}_15_F1.txt"
      sample_tag="${sample}_15_F1"
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcremp_${sample_tag}
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/${sample_tag}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=short

set -euo pipefail

tcremp-run -i "$sample_file" -c TRB -o "$EMB_DIR" -np 32
EOF
  done
done
