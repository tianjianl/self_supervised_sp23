#CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task rte --lr 0.000001 --epoch 10 #57
#CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task rte --lr 0.00001 --epoch 10  --use_sd --sd_alpha 0.3 --student_layer 12
#CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task rte --lr 0.00001 --epoch 10  --use_sd --sd_alpha 0.5 --student_layer 12
#CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task rte --lr 0.00001 --epoch 10  --use_sd --sd_alpha 0.7 --student_layer 12
CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task rte --seed 12 --lr 0.00001 --epoch 10 #57
CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task rte --seed 42 --lr 0.00001 --epoch 10 #57
CUDA_VISIBLE_DEVICES=$1 python3 finetune_opt.py --task rte --seed 33 --lr 0.00001 --epoch 10 #57
