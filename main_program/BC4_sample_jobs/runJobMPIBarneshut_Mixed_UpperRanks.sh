#!/bin/bash
#SBATCH --job-name=runJobMPIBarneshut
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033184
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --time=0:6:00
#SBATCH --mem=100M

# Load modules
module add languages/python/3.12.3
cd $SLURM_SUBMIT_DIR

# Run Program
mpirun -np 8 python barneshut_core_MPI_py.py 5 1000 1
mpirun -np 12 python barneshut_core_MPI_py.py 5 1000 1
mpirun -np 20 python barneshut_core_MPI_py.py 5 1000 1
