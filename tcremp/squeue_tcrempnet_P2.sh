#!/bin/sh

#SBATCH --job-name=tcrempnet_P2        # Job name
#SBATCH --cpus-per-task=16        # Run on a single CPU
#SBATCH --mem=256gb                 # Job memory request
#SBATCH --time=02:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=short


python tcrEmpNet.py \
  --sample /projects/immunestatus/pogorelyy/airr_format/P2_15_F1.txt \
  --background /projects/immunestatus/pogorelyy/airr_format/P2_0_F1_with_1.txt \
  --output /projects/immunestatus/pogorelyy/tcrempnet/P2_F1 \
  --chain TRB \
  --prefix yfv_P2_F1_with_1 \
  -np 100