#!/usr/bin/env bash
# MSP Energy GradNorm ODIN
MODEL=$1
METHOD=$2
BATCH=$3

python test_ood.py \
      --name test_${METHOD}_sweep \
      --in_datadir /dataset/cv/imagenet/val \
      --out_dataroot /dataset/cv/ood_data \
      --batch $BATCH \
      --model $MODEL\
      --logdir checkpoints/$MODEL/test_${METHOD}_sweep \
      --score $METHOD