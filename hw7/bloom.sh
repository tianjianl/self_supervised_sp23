#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --job-name="HW7 CS 601.471/671 homework"
#SBATCH --output=bloomz-few-shot.out

module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

#conda activate toy_classification_env # open the Python environment

# pip install -r requirements.txt # install Python dependencies

# runs your code
srun python bloom.py
