#!/bin/sh

#SBATCH --job-name=tcremp_as_Mikh        # Job name
#SBATCH --cpus-per-task=48        # Run on a single CPU
#SBATCH --mem=16gb                 # Job memory request
#SBATCH --time=08:00:00           # Time limit hrs:min:sec
#SBATCH --output=tcremp_as_Mikh.%j.log   # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizaveta.k.vlasova@gmail.com
#SBATCH --constraint=hpc
#SBATCH --partition=medium


tcremp-run \
  --input /projects/immunestatus/pogorelyy/airr_format/P1_0_F1_with_1.txt \
  --output /projects/immunestatus/test \
  --chain TRB \
  -np 48
