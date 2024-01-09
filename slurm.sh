#!/bin/bash
#SBATCH --job-name=rnn
#SBATCH --output=logs/slurm/rnn.%A.%a.out
#SBATCH --error=logs/slurm/rnn.%A.%a.err
#SBATCH --partition=shared-cpu
#SBATCH --array=1-180
#SBATCH --mem=7000
#SBATCH --time=12:00:00
# 1300/4=325

# source /home/users/f/findling/.bash_profile
# mamba activate iblenv

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
python rnn.py $SLURM_ARRAY_TASK_ID
