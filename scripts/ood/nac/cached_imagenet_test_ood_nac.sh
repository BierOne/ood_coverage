#!/bin/bash
# sh scripts/ood/ash/imagenet_test_ood_ash.sh
MODEL=$1
BATCH=$2
SubNum=$3
LAYER=$4
APS=$5
AUG=$6

# available architectures:
# resnet50, swin-t, vit-b-16
# ood

if [[ "aps" = "$APS" ]]; then
python scripts/eval_ood_imagenet.py \
      --tvs-pretrained \
      --arch $MODEL \
      --aug $AUG \
      --postprocessor nac \
      --batch-size $BATCH \
      --valid-num $SubNum \
      --layer-names $LAYER \
      --save-score --save-csv --aps --use_cache #--fsood
else
  python scripts/eval_ood_imagenet.py \
      --tvs-pretrained \
      --arch $MODEL \
      --aug $AUG \
      --postprocessor nac \
      --batch-size $BATCH \
      --valid-num $SubNum \
      --layer-names $LAYER \
      --save-score --save-csv --use_cache #--fsood

fi

