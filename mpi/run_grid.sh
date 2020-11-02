#!/bin/bash
#SBATCH -J orbit-grid
#SBATCH -o logs/run_grid.o%j
#SBATCH -e logs/run_grid.e%j
#SBATCH -N 16
#SBATCH -t 06:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/nonlinear-dynamics-fun/scripts

date

mpirun -n $SLURM_NTASKS python3 run_grid.py --mpi

date
