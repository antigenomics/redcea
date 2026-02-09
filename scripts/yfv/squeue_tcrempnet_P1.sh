#!/bin/sh

#SBATCH --job-name=tcrempnet_P1       # Job name
#SBATCH --cpus-per-task=48     # Run on a single CPU
#SBATCH --mem=256gb                 # Job memory request
#SBATCH --time=08:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=medium


tcrempnet \
  --sample /projects/immunestatus/pogorelyy/airr_format/P1_15_F1.txt \
  --background /projects/immunestatus/pogorelyy/airr_format/P1_0_F1_with_1.txt \
  --output /projects/immunestatus/pogorelyy/tcrempnet/P1_F1_test_old \
  --chain TRB \
  --prefix yfv_P1_F1_with_1 \
  -np 48