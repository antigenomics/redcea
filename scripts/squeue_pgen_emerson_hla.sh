#!/bin/sh

#SBATCH --job-name=olga_emerson       # Job name
#SBATCH --cpus-per-task=100     # Run on a single CPU
#SBATCH --mem=64gb                 # Job memory request
#SBATCH --time=1:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=short


python pgens_compute.py --name emerson_hla --processes 100