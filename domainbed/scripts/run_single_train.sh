#  VLCS PACS OfficeHome DomainNet (6) ColoredMNIST (3) RotatedMNIST (6) TerraIncognita
#  ERM SelfReg MMD Mixup CORAL RSC Fish IRM DANN
#  CDANN MLDG GroupDRO ARM MTL VREx ANDMask SANDMask
#  SagNet Fishr CondCAD

METHOD=$1
DATASET=$2
MODEL=$3

### add your config here
if [[ "ERM" = "$METHOD" ]]; then
  config_name="erm_config.yaml"
else
  config_name="erm_config.yaml"
fi


for tseed in 1; do
if [[ "DomainNet" = "$1" ]]; then
python train_all.py $1_T${tseed} param_config/${config_name}\
       --trial_seed $tseed\
       --data_dir=/dg/data/domainbed/\
       --algorithm $METHOD\
       --dataset $DATASET\
       --single_train\
       --model $MODEL\
       --test_batchsize 100
else
python train_all.py $1_T${tseed} param_config/${config_name}\
       --trial_seed $tseed\
       --data_dir=/dg/data/domainbed/\
       --algorithm $METHOD\
       --dataset $DATASET\
       --test_batchsize 100\
       --single_train\
       --model $MODEL
fi
done