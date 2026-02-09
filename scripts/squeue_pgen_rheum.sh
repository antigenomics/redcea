#!/bin/sh

#SBATCH --job-name=olga_rheum       # Job name
#SBATCH --cpus-per-task=48     # Run on a single CPU
#SBATCH --mem=232gb                 # Job memory request
#SBATCH --time=8:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=medium


python pgens_compute.py --name rheum --processes 48