export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVIDES=6,7,8,9 accelerate launch continue_pretrain_bert.py
