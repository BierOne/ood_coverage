#!/bin/bash
# sh scripts/ood/mds/cifar100_test_ood_mds.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/mds.yml \
    --network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
    --id-data cifar100 \
    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor mds \
    --save-score --save-csv
