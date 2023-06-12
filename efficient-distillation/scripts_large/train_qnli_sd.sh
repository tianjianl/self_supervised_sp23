#python3 finetune_bert.py --task qnli --use_sd --student_layer 12 --lr 0.000005 --bs 16 --seed 42 --epoch 10 --max_len 256 
#python3 finetune_bert.py --task qnli --use_sd --student_layer 10 --lr 0.000005 --bs 16 --seed 42 --epoch 10 --max_len 256 
#python3 finetune_bert.py --task qnli --use_sd --student_layer 8 --lr 0.000005 --bs 16 --seed 42 --epoch 10 --max_len 256 
python3 finetune_bert.py --task qnli --use_sd --student_layer 6 --lr 0.000005 --bs 16 --seed 42 --epoch 10 --max_len 256 
