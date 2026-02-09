#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/vdjdb/airr_format"
EMB_DIR="/projects/immunestatus/vdjdb/tcremp"

# Список сэмплов
samples=(
trb_vdjdb_uni_AVFDRKSDAK trb_vdjdb_uni_ELAGIGILTV trb_vdjdb_uni_GILGFVFTL 
trb_vdjdb_uni_GLCTLVAML trb_vdjdb_uni_NLVPMVATV trb_vdjdb_uni_RAKFKQLL 
trb_vdjdb_uni_SLLMWITQV trb_vdjdb_uni_YLQPRTFLL trb_vdjdb_uni_YVLDHLIVV
)

# Запуск цикла сабмита задач
for sample in "${samples[@]}"; do
  SAMPLE_FILE="$AIRR_DIR/${sample}.tsv"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcremp_${sample}
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=logs_tcremp/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=short

tcremp-run -i "$SAMPLE_FILE" -c TRB -o "$EMB_DIR" -np 32 -kn 12
EOF

done
