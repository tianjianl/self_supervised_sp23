# This hyper-parameter configuration is from https://huggingface.co/gchhablani/bert-large-cased-finetuned-cola
python3 finetune_bert.py --task cola --lr 0.000003 --bs 4 --seed 42 --epoch 10 --max_len 256 


