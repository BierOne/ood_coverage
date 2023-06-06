# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torchvision.models
# import timm
import os,sys
from torchvision import transforms


def get_transform(name, train=False):
    if "bit" in name.lower():
        imagesize = 480
        transform_test = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif ("resnet" in name.lower()) or ("mobilenet" in name.lower()):
        imagesize = 224
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(name)

    input_shape = (1, 3, imagesize, imagesize)
    if train:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform_train = transform_test

    return input_shape, transform_train, transform_test


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def torchhub_load(repo, model, **kwargs):
    try:
        # torch >= 1.10
        network = torch.hub.load(repo, model=model, skip_validation=True, **kwargs)
    except TypeError:
        # torch 1.7.1
        network = torch.hub.load(repo, model=model, **kwargs)

    return network


def get_backbone(name, preserve_readout=True, num_classes=1000, pretrained=True, **kwargs):
    if name == "resnet18":
        network = torchvision.models.resnet18(pretrained=pretrained)
        n_outputs = 512
    elif name == "resnet50":
        from models.resnet import resnet50
        network = resnet50(num_classes=num_classes, pretrained=True, **kwargs)
        # network = torchvision.models.resnet50(pretrained=pretrained)
        n_outputs = 2048

    elif name == "resnet101":
        network = torchvision.models.resnet101(pretrained=pretrained)
        n_outputs = 2048

    elif name == "mobilenet":
        from models.mobilenet import mobilenet_v2
        network = mobilenet_v2(num_classes=num_classes, pretrained=True, **kwargs)
        n_outputs = 1280

    elif name == "BiT-S-R101x1":
        from models import resnetv2
        network = resnetv2.KNOWN_MODELS[name](head_size=num_classes, **kwargs)  # ImageNet pretrain
        f_path = os.path.join("./checkpoints/pretrained", "BiT-S-R101x1-flat-finetune.pth.tar")
        state_dict = torch.load(f_path)
        network.load_state_dict_custom(state_dict['model'])
        n_outputs = 2048

    else:
        raise ValueError(name)

    if not preserve_readout:
        # remove readout layer (but left GAP and flatten)
        # final output shape: [B, n_outputs]
        if name.startswith("resnet"):
            del network.fc
            network.fc = Identity()
        elif name.startswith("vit"):
            del network.head
            network.head = Identity()

    return network, n_outputs
