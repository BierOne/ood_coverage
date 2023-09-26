#!/bin/bash
# sh scripts/ood/ash/imagenet_test_ood_ash.sh


CKPT=$1

if [[ "augmix" = "$CKPT" ]]; then
  CKPT=./results/imagenet_resnet50_tvsv1_augmix_default/ckpt.pth
elif [[ "pixmix" = "$CKPT" ]]; then
  CKPT=./results/imagenet_resnet50_tvsv1_base_pixmix/ckpt.pth
elif [[ "regmix" = "$CKPT" ]]; then
  CKPT=./results/imagenet_resnet50_regmixup_e30_lr0.001_alpha10_default/s0/best.ckpt
elif [[ "deepaug" = "$CKPT" ]]; then
  CKPT=./results/imagenet_resnet50_tvsv1_base_deepaugment/ckpt.pth
elif [[ "style" = "$CKPT" ]]; then
  CKPT=./results/imagenet_resnet50_tvsv1_base_stylized/ckpt.pth
elif [[ "randaug" = "$CKPT" ]]; then
  CKPT=./results/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt
fi


############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50, swin-t, vit-b-16
# ood
python scripts/eval_ood_imagenet.py \
    --arch resnet50 \
    --postprocessor ash \
    --ckpt-path $CKPT \
    --save-score --save-csv #--fsood

# full-spectrum ood
#python scripts/eval_ood_imagenet.py \
#    --tvs-pretrained \
#    --arch resnet50 \
#    --postprocessor ash \
#    --save-score --save-csv --fsood
