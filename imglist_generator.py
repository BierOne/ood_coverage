import os
import argparse
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--root', default="./data/benchmark_imglist")
parser.add_argument('--split_num', default=1000, type=int, help="number of training data samples for sanity check")
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_args()

if args.dataset == "cifar10":
    fname = "train_cifar10.txt"
elif args.dataset == "cifar100":
    fname = "train_cifar100.txt"
elif args.dataset == "imagenet":
    fname = "train_imagenet.txt"
elif args.dataset == "imagenet200":
    fname = "train_imagenet200.txt"
else:
    raise ValueError(args.dataset)

fpath = os.path.join(args.root, args.dataset, fname)
cls_to_imgs = defaultdict(list)
with open(fpath, 'r') as imgfile:
    imglist = imgfile.readlines()
    for img_idx in range(len(imglist)):
        line = imglist[img_idx].strip('\n')
        tokens = line.split(' ', 1)
        if tokens[0].startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        _, label = tokens
        cls_to_imgs[label].append(imglist[img_idx])

# keep class balance
num_each_cls = int(args.split_num / len(cls_to_imgs))
sub_imgs, main_imgs = [], []
for cls in cls_to_imgs:
    imgs = cls_to_imgs[cls]
    np.random.RandomState(args.seed).shuffle(imgs)
    sub_imgs += imgs[:num_each_cls]
    main_imgs += imgs[num_each_cls:]

for name, img_tokens in zip(["sub_" + fname, "main_" + fname],
                            [sub_imgs, main_imgs]):
    fpath = os.path.join(args.root, args.dataset, name)
    print("write lists to: ", fpath)
    print("num of images: ", len(img_tokens))
    with open(fpath, 'w') as f:
        for line in img_tokens:
            f.write(line)
        f.close()
