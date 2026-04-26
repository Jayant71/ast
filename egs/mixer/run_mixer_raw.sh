#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TORCH_HOME=../../pretrained_models

model=ast
dataset=mixer
imagenetpretrain=True
audiosetpretrain=False
bal=none
lr=1e-4
freqm=24
timem=282
mixup=0
epoch=50
batch_size=2
fstride=10
tstride=16

audio_length=3000
noise=False
skip_norm=True
dataset_mean=0
dataset_std=1

metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

n_class=6

timestamp=$(date +%Y%m%d_%H%M%S)
base_exp_dir=./exp/test-${dataset}-raw-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}-${timestamp}

tr_data=./data/mixer_train_data_processed.json
te_data=./data/mixer_eval_data_processed.json

if [ ! -f "$tr_data" ] || [ ! -f "$te_data" ]; then
    echo "ERROR: Data files not found."
    echo "Run first: python generate_data_files.py --dataset_path /path/to/dataset"
    exit 1
fi

if [ -d "$base_exp_dir" ]; then
    echo "Experiment directory $base_exp_dir already exists. Remove it to retrain."
    exit 1
fi
mkdir -p "$base_exp_dir"

echo "Training mixer classification model (raw audio, no preprocessing, no normalization)"

exp_dir=${base_exp_dir}

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir "$exp_dir" \
--label-csv ./data/mixer_class_labels_indices.csv --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} \
--noise ${noise} --skip_norm ${skip_norm}

echo "Training completed. Results saved in ${exp_dir}"

python ./get_mixer_result.py --exp_path "$exp_dir"
