#conda activate myenv # open the Python environment
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES=6,7,8,9 accelerate launch continue_pretrain_bert.py --student_layer 12 --add_student
