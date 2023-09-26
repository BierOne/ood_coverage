#!/bin/bash
# sh scripts/ood/gradnorm/cifar100_test_ood_gradnorm.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

#PYTHONPATH='.':$PYTHONPATH \
## srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
## --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
## --kill-on-bad-exit=1 --job-name=${jobname} \
#
#python main.py \
#    --config configs/datasets/cifar100/cifar100.yml \
#    configs/datasets/cifar100/cifar100_ood.yml \
#    configs/networks/resnet18_32x32.yml \
#    configs/pipelines/test/test_ood.yml \
#    configs/preprocessors/base_preprocessor.yml \
#    configs/postprocessors/gradnorm.yml \
#    --num_workers 8 \
#    --network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
#    --mark 0

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
    --id-data cifar100 \
    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor gradnorm \
    --save-score --save-csv
