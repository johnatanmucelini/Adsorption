#!/bin/bash
#SBATCH -t 20-00 -n 1

module load orca
goo-job-nanny srun -n 1 $ORCA_BIN Acrylic_acid.inp
