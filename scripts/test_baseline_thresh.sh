#!/usr/bin/env bash
# MSP Energy GradNorm ODIN
MODEL=$1
METHOD=$2
BATCH=$3

if [[ "resnet50" = "$MODEL" ]]; then
  thresh=1.0 # default best value
elif [[ "mobilenet" = "$MODEL" ]]; then
  thresh=1.0  # default best value
else
  thresh=2.0 # default best value
fi


python test_ood.py \
      --use-thresh\
      --thresh $thresh\
      --name test_${METHOD}_sweep \
      --in_datadir /dataset/cv/imagenet/val \
      --out_dataroot /dataset/cv/ood_data \
      --batch $BATCH \
      --model $MODEL\
      --logdir checkpoints/$MODEL/test_thresh_${thresh}_${METHOD}_sweep \
      --score $METHOD