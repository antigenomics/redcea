#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/emerson_hla/airr_format"
EMB_DIR="/projects/immunestatus/emerson_hla/tcremp"

# Список сэмплов
samples=(
joint_non_HLA-A02 joint_non_HLA-A03 joint_non_HLA-A24
)

# Запуск цикла сабмита задач
for sample in "${samples[@]}"; do
  SAMPLE_FILE="$AIRR_DIR/${sample}.tsv"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcremp_${sample}
#SBATCH --cpus-per-task=64
#SBATCH --mem=128gb
#SBATCH --time=06:00:00
#SBATCH --output=logs_tcremp/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=medium

tcremp-run -i "$SAMPLE_FILE" -c TRB -o "$EMB_DIR" -np 64
EOF

done
