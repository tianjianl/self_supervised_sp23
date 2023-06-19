CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task cola --lr 0.000001 --epoch 10 --max_len 256 --bs 8
