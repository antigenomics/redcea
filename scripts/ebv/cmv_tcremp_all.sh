#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/emerson/airr_format"
EMB_DIR="/projects/immunestatus/emerson/tcremp"

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

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcrempnet_${sample}
#SBATCH --cpus-per-task=100
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH --output=logs_tcremp/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=medium

tcremp-run -i "$SAMPLE_FILE" -c TRB -o "$EMB_DIR" -np 100
EOF

done
