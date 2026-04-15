#!/bin/tcsh
##### These lines are for Slurm
#SBATCH -N 1
#SBATCH -J rosa runtimes
#SBATCH -t 12:00:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morgajoa@oregonstate.edu

##### These are shell commands
date
echo ""
lscpu
echo "" 

module load rocm/6
module list
source ~/dane/bin/activate

echo ""
echo ""
echo ""

python runtime.py

echo ""
echo ""
echo "*****JOB COMPLETE******"
echo ""
echo ""