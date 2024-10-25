#  VLCS PACS OfficeHome DomainNet (6) ColoredMNIST (3) RotatedMNIST (6) TerraIncognita
#  ERM SelfReg MMD Mixup CORAL RSC Fish IRM DANN
#  CDANN MLDG GroupDRO ARM MTL VREx ANDMask SANDMask
#  SagNet Fishr CondCAD

METHOD=$1
DATASET=$2
MODEL=$3

if [[ "ERM" = "$METHOD" ]]; then
  config_name="erm_config.yaml"
else
  config_name="erm_config.yaml"
fi


for tseed in 0; do
if [[ "DomainNet" = "$1" ]]; then
python train_all.py $1_T${tseed} param_config/${config_name}\
       --trial_seed $tseed\
       --data_dir=/dg/data/domainbed/\
       --algorithm $METHOD\
       --dataset $DATASET\
       --prebuild_loader\
       --model $MODEL\
       --test_batchsize 64
else
python train_all.py $1_T${tseed} param_config/${config_name}\
       --trial_seed $tseed\
       --data_dir=/dg/data/domainbed/\
       --algorithm $METHOD\
       --dataset $DATASET\
       --test_batchsize 64\
       --model $MODEL\
       --save_model_every_checkpoint
fi
done