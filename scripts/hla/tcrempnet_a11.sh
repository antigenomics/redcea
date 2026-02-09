#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/emerson_hla/airr_format"
EMB_DIR="/projects/immunestatus/emerson_hla/tcremp"
OUT_DIR="/projects/immunestatus/emerson_hla/tcrempnet"
BACKGROUND="$AIRR_DIR/joint_non_HLA-A11.tsv"
BACKGROUND_EMB="$EMB_DIR/joint_non_HLA-A11_embeddings.parquet"

# Список сэмплов
samples=(
HIP14074 HIP14209 HIP00825 HIP14911 HIP12143
)

# Запуск цикла сабмита задач
for sample in "${samples[@]}"; do
  SAMPLE_FILE="$AIRR_DIR/${sample}.tsv"
  SAMPLE_EMB="$EMB_DIR/${sample}_embeddings.parquet"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcrempnet_${sample}
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=6:00:00
#SBATCH --output=logs_tcrempnet/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=medium

tcrempnet \\
  -is "$SAMPLE_FILE" \\
  -ib "$BACKGROUND" \\
  -c TRB -o "$OUT_DIR" -np 16 \\
  -se "$SAMPLE_EMB" \\
  -be "$BACKGROUND_EMB"
EOF

done
