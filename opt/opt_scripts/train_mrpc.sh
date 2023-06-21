CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task mrpc --use_sd --student_layer 12 --lr 0.000005 --epoch 10 --max_len 512 --bs 4 --sd_alpha 0.5

CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task mrpc --use_sd --student_layer 12 --lr 0.000005 --epoch 10 --max_len 512 --bs 4 --sd_alpha 0.7
CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task mrpc --use_sd --student_layer 12 --lr 0.000005 --epoch 10 --max_len 512 --bs 4 --sd_alpha 0.3
