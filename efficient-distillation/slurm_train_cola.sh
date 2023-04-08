#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="bert-large-cola"
#SBATCH --output="bert-large-cola.txt"
#SBATCH --mem=20G

module load anaconda

# init virtual environment if needed
conda create -n myenv python=3.7

conda activate myenv # open the Python environment
pip3 install -r requirements.txt

srun python3 finetune_bert.py --task cola --use_sd --student_layer 12 --lr 0.000003 --bs 8 --seed 42 --epoch 10 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --student_layer 11 --lr 0.000003 --bs 8 --seed 42 --epoch 10 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --student_layer 10 --lr 0.000003 --bs 8 --seed 42 --epoch 10 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --student_layer 9 --lr 0.000003 --bs 8 --seed 42 --epoch 10 --max_len 256 
srun python3 finetune_bert.py --task cola --use_sd --student_layer 8 --lr 0.000003 --bs 8 --seed 42 --epoch 10 --max_len 256 
