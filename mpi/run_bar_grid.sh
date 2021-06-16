#!/bin/bash
#SBATCH -J bar-grid
#SBATCH -o logs/bar_grid.o%j
#SBATCH -e logs/bar_grid.e%j
#SBATCH -N 8
#SBATCH -t 04:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

source ~/.bash_profile
init_conda

cd /mnt/ceph/users/apricewhelan/projects/nonlinear-dynamics-fun/scripts

date

mpirun python3 -m mpi4py.run -rc thread_level='funneled' \
bar_chaos_run.py --mpi

date
