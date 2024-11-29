#!/bin/bash
#SBATCH --job-name=runJobMPIBarneshut
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033184
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --mem=100M

# Load modules
module add languages/python/3.12.3
cd $SLURM_SUBMIT_DIR

# Run Program
mpirun -np 2 python barneshut_core_MPI_py.py 5 1000 1
mpirun -np 3 python barneshut_core_MPI_py.py 5 1000 1
mpirun -np 4 python barneshut_core_MPI_py.py 5 1000 1
mpirun -np 5 python barneshut_core_MPI_py.py 5 1000 1
mpirun -np 6 python barneshut_core_MPI_py.py 5 1000 1
mpirun -np 7 python barneshut_core_MPI_py.py 5 1000 1
