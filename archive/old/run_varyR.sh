#!/bin/sh
#SBATCH --account=theory      # The account name for the job.
#SBATCH -c 4                     # The number of cpu cores to use.
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.
#SBATCH --output=PlasticNet_R=%a_w2only_w1plusminus.out
#SBATCH --array=1-10,15,30

stdbuf -oL python script_varyR.py $SLURM_ARRAY_TASK_ID
