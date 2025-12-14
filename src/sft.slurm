#!/bin/bash

#SBATCH --job-name=ft_base
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=90GB
#SBATCH --time=0-06:00:00
#SBATCH --array=16-22
#SBATCH --output=/scratch/craj/langsense/logs/slurm_logs/finetuning/out/ft_base.%A_%a.out.txt
#SBATCH --error=/scratch/craj/langsense/logs/slurm_logs/finetuning/err/ft_base.%A_%a.err.txt

#nvidia-smi
#cd /scratch/craj/diy/src/3_experiments

# Read arguments from the config file
readarray -t ARGS < /scratch/craj/langsense/src/sft.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

#echo $ARGUMENTS
echo "Task $SLURM_ARRAY_TASK_ID → $ARGUMENTS"

python sft.py $ARGUMENTS
