#!/bin/bash

set -e
set -x
#22.9 CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task cola --lr 0.000001 --epoch 10 --max_len 256 --bs 8

CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task cola --lr 0.0000003 --epoch 10 --max_len 256 --bs 8
CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task cola --lr 0.0000007 --epoch 10 --max_len 256 --bs 8

CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task cola --lr 0.0000003 --epoch 10 --max_len 256 --bs 8 --use_sd
CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task cola --lr 0.0000007 --epoch 10 --max_len 256 --bs 8 --use_sd
