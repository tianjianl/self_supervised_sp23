#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="bert-base-cola"
#SBATCH --output="bert-base-cola-alpha.txt"
#SBATCH --mem=20G

module load anaconda

# init virtual environment if needed

conda activate myenv # open the Python environment

srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 8 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 7 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 6 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 5 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 4 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
