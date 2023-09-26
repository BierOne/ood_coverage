NUM=$1

# we split 1000 training images for sanity check
python imglist_generator.py --dataset cifar10 --split_num $NUM --seed 1;
python imglist_generator.py --dataset cifar100 --split_num $NUM --seed 1;
python imglist_generator.py --dataset imagenet200 --split_num $NUM --seed 1;
python imglist_generator.py --dataset imagenet --split_num $NUM --seed 1