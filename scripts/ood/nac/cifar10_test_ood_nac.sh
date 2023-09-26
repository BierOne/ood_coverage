#!/bin/bash
# sh scripts/ood/ash/cifar10_test_ood_ash.sh
BATCH=$1
SubNum=$2
LAYER=$3
APS=$4

if [[ "aps" = "$APS" ]]; then
  python scripts/eval_ood.py \
      --id-data cifar10 \
      --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
      --postprocessor nac \
      --batch-size $BATCH \
      --valid-num $SubNum \
      --layer-names $LAYER \
      --save-score --aps --save-csv
else
  python scripts/eval_ood.py \
      --id-data cifar10 \
      --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
      --postprocessor nac \
      --batch-size $BATCH \
      --valid-num $SubNum \
      --layer-names $LAYER \
      --save-score --save-csv
fi
