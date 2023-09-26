#!/bin/bash
# sh scripts/ood/godin/cifar100_train_godin.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/godin_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/godin.yml \
    --network.backbone.name resnet18_32x32 \
    --num_workers 8 \
    --trainer.name godin \
    --optimizer.num_epochs 100 \
    --merge_option merge \
    --seed 0
