#!/bin/sh
#SBATCH --account=theory      
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16gb        
#SBATCH --output=PlasticRecall_%x_R=%a.out
#SBATCH --array=1,2,5

echo $SLURM_ARRAY_TASK_ID 
echo $SLURM_JOB_NAME
echo $SLURM_JOB_ID

stdbuf -oL python script_delayed_recall.py $SLURM_ARRAY_TASK_ID $SLURM_JOB_NAME
