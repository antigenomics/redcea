#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/emerson/airr_format"
EMB_DIR="/projects/immunestatus/emerson/tcremp"
OUT_DIR="/projects/immunestatus/emerson_ebv/tcrempnet"
BACKGROUND="$AIRR_DIR/control_ebv.tsv"
BACKGROUND_EMB="$EMB_DIR/control_ebv_embeddings.parquet"

# Список сэмплов
samples=(
HIP00594  HIP00777  HIP03370  HIP03511  HIP03720  
HIP04509  HIP05559  HIP05815  HIP05960  HIP08653  
HIP08986  HIP09020  HIP10716  HIP11937  HIP12900  
HIP13513  HIP13903  HIP13967  HIP13986  HIP13994  
HIP14041  HIP14051  HIP14157  HIP14194  HIP14227  
HIP15854
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
#SBATCH --time=2:00:00
#SBATCH --output=logs_tcrempnet/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=short

tcrempnet \\
  -is "$SAMPLE_FILE" \\
  -ib "$BACKGROUND" \\
  -c TRB -o "$OUT_DIR" -np 16 \\
  -se "$SAMPLE_EMB" \\
  -be "$BACKGROUND_EMB"
EOF

done
