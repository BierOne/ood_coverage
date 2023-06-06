#!/usr/bin/env bash

METHOD=Coverage
MODEL=$1
BATCH=$2
if [[ "resnet50" = "$MODEL" ]]; then
  LAYER="avgpool"
elif [[ "mobilenet" = "$MODEL" ]]; then
  LAYER="avgpool"
elif [[ "BiT-S-R101x1" = "$MODEL" ]]; then
  LAYER="before_head"
fi

python test_ood.py \
      --seed -1\
      --name test_${METHOD}_sweep \
      --in_datadir /dataset/cv/imagenet/val \
      --out_dataroot /dataset/cv/ood_data_arxiv \
      --batch $BATCH \
      --layer-name $LAYER\
      --model $MODEL\
      --logdir checkpoints/$MODEL/$LAYER \
      --score ${METHOD}

