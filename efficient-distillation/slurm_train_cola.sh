#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name="bert-large-sst-2-8"
#SBATCH --output="bert-large-sst-2-new-8.txt"
#SBATCH --mem=20G

module load anaconda

# init virtual environment if needed
#conda create -n myenv python=3.7

wandb login

conda activate myenv # open the Python environment

<<<<<<< HEAD
srun python3 finetune_bert.py --task sst-2 --use_sd --sd_alpha 0.3 --model bert-base-uncased --student_layer 8 --lr 0.000003 --bs 8 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task sst-2 --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 8 --lr 0.000003 --bs 8 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task sst-2 --use_sd --sd_alpha 0.7 --model bert-base-uncased --student_layer 8 --lr 0.000003 --bs 8 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task sst-2 --use_sd --sd_alpha 0.9 --model bert-base-uncased --student_layer 8 --lr 0.000003 --bs 8 --seed 42 --epoch 20 --max_len 256 
=======
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 8 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 7 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 6 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 5 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --sd_alpha 0.5 --model bert-base-uncased --student_layer 4 --lr 0.000003 --bs 16 --seed 42 --epoch 20 --max_len 256 
>>>>>>> fced983417cae5eeddc00775ae783cb4f5b07652
