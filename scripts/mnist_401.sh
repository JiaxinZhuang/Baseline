set -e
set -x
export PYTHONPATH='src'

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}

cuda=5
dataset=mnist
backbone=NIN
n_epochs=400
batch_size=128
optimizer=SGD
re_size=28
input_size=28
test_input_size=28
eval_frequency=10
server=ls15
learning_rate=1e-4

CUDA_VISIBLE_DEVICES=$cuda python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --cuda=$cuda \
    --dataset=$dataset \
    --n_epochs=$n_epochs \
    --batch_size=$batch_size \
    --optimizer=$optimizer \
    --eval_frequency=$eval_frequency \
    --input_size=$input_size \
    --re_size=$re_size \
    --backbone=$backbone \
    --server=$server \
    --learning_rate=$learning_rate \
    --test_input_size=$test_input_size \
    --resume=0 \
    2>&1 | tee $log_file
