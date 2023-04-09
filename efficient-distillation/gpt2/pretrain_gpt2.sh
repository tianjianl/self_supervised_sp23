#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="gpt2-wikitext"
#SBATCH --output="gpt2-wikitext.log"
#SBATCH --mem=50G
#SBATCH --mail-user=tli104@jhu.edu  
module load anaconda

# init virtual environment if needed
conda create -n myenv python=3.7

conda activate myenv # open the Python environment
pip install git+https://github.com/huggingface/transformers

DATA_DIR="/scratch4/cs601/tli104"
MODEL_DIR="/scratch4/cs601/tli104/gpt2/checkpoints"

mkdir -p $MODEL_DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch run_clm_no_trainer.py \
  --model_name_or_path gpt2-large \
  --train_file $DATA_DIR/wikitext-103/wiki.train.tokens.txt \
  --validation_file $DATA_DIR/wikitext-103/wiki.valid.tokens.txt \
  --output_dir $MODEL_DIR \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --checkpointing_steps 4096 \
