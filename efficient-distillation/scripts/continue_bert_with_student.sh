#!/bin/bash

#SBATCH -A danielk80_gpu
#SBATCH --partition ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name="bert-continue-teacher-student"
#SBATCH --output="bert-continue-log.txt"
#SBATCH --mem=50G
#SBATCH --mail-user=tli104@jhu.edu  
#SBATCH --mail-type=ALL

module load anaconda

# init virtual environment if needed

conda activate myenv # open the Python environment
export TORCH_DISTRIBUTED_DEBUG=DETAIL
accelerate launch continue_pretrain_bert.py --student_layer 12 --add_student
