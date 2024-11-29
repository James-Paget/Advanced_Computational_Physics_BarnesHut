#!/bin/bash
#SBATCH --job-name=runJobOpenMpBarneshut
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033184
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:10:00
#SBATCH --mem=100M
#SBATCH --output=slurm-openMp.out

# Load modules
module add languages/python/3.12.3
cd $SLURM_SUBMIT_DIR

# Run Program
python barneshut_core_OpenMP_py.py 5 1 10000 1
