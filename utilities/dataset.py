"""
Create Time: 23/4/2023
Author: BierOne (lyibing112@gmail.com)
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import numpy as np

def pil_loader(img_str, str='RGB'):
    with Image.open(img_str) as img:
        img = img.convert(str)
    return img

def get_dataset_for_bit(datadir, transform=None):
    root_dir = datadir.split('/')
    meta_type = root_dir[-1]  # train or val
    root_dir = '/'.join(root_dir[:-1])
    if meta_type == "train":
        return DatasetWithMeta(root_dir, "data_lists/imagenet2012_train_list.txt", transform)
    elif meta_type == "val":
        return DatasetWithMeta(root_dir, "data_lists/imagenet2012_val_list.txt", transform)
    else:
        raise ValueError(meta_type)

class DatasetWithMeta(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        super(DatasetWithMeta, self).__init__()
        if ('train' in root_dir) or ('val' in root_dir):
            root_dir = root_dir.split('/')
            root_dir = '/'.join(root_dir[:-1])
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        self.classes = set()

        self.samples = []
        for line in lines:
            segs = line.strip().split(' ')
            self.samples.append((os.path.join(self.root_dir, ' '.join(segs[:-1])), int(segs[-1])))
            self.classes.add(int(segs[-1]))
        self.num = len(self.samples)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename, cls_id = self.samples[idx]
        try:
            img = pil_loader(filename)
        except:
            print(filename)
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls_id







from collections import defaultdict

class Subset(Dataset):
    def __init__(self, dataset, valid_ratio=0.05):
        # indices = torch.randperm(len(dataset))[:limit]
        # self.indices = indices
        self.dataset = dataset
        self.classes = dataset.classes

        self.cls_to_idxes = defaultdict(list)
        for idx, entry in enumerate(dataset.samples):
            self.cls_to_idxes[entry[1]].append(idx)
        self.set_sub_samples_on(valid_ratio)

    def set_sub_samples_on(self, ratio=0.05):
        self.use_valid_inds = True
        self.valid_inds = []
        min_len = min([len(inds) for inds in self.cls_to_idxes.values()])
        valid_num = round(min_len * ratio)
        for cls in self.cls_to_idxes:
            inds = self.cls_to_idxes[cls]
            np.random.RandomState().shuffle(inds)
            self.valid_inds += inds[:valid_num]

    def set_sub_samples_off(self):
        self.use_valid_inds = False

    def __getitem__(self, idx):
        if self.use_valid_inds:
            idx = self.valid_inds[idx]
        return self.dataset[idx]

    def __len__(self):
        if self.use_valid_inds:
            return len(self.valid_inds)
        return len(self.dataset)
