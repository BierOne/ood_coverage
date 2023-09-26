"""
Create Time: 22/3/2023
Author: BierOne (lyibing112@gmail.com)
"""
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path


def clear_path(dir, pattern="state_statistics_step*.pkl"):
    for p in Path(dir).glob(pattern):
        p.unlink()


def sigmoid(x, sig_alpha=1.0):
    """
    sig_alpha is the steepness controller (larger denotes steeper)
    """
    return 1 / (1 + torch.exp(-sig_alpha * x))


from torch.utils.data import Dataset
from collections import defaultdict


class TrainSubset(Dataset):
    def __init__(self, dataset, valid_ratio=None, valid_num=None,
                 balanced=True, rand_seed=1):
        assert (valid_num is None) ^ (valid_ratio is None)
        self.dataset = dataset
        self.cls_to_idxes = defaultdict(list)
        for idx, entry in enumerate(dataset.samples):
            self.cls_to_idxes[entry[1]].append(idx)

        self.rand_seed = rand_seed
        self.get_sub_samples(valid_ratio, valid_num, balanced)
        self.set_sub_samples_on()

    def get_sub_samples(self, ratio=None, valid_num=None, balanced=None):
        self.use_sub_inds = True
        self.valid_inds = []
        self.remain_inds = []
        if balanced:
            min_len = min([len(inds) for inds in self.cls_to_idxes.values()])
            if ratio is not None:
                sub_num = int(min_len * ratio)
            elif valid_num is not None:
                sub_num = int(valid_num / len(self.cls_to_idxes))
                sub_num = min_len if sub_num > min_len else sub_num
                print("number for each class:", sub_num)

            for cls in self.cls_to_idxes:
                inds = self.cls_to_idxes[cls]
                np.random.RandomState(self.rand_seed).shuffle(inds)
                self.valid_inds += inds[:sub_num]
                self.remain_inds += inds[sub_num:]
        else:
            if valid_num is None:
                valid_num = round(ratio * len(self.dataset.samples))
            inds = list(range(len(self.dataset)))
            np.random.RandomState(self.rand_seed).shuffle(inds)
            self.valid_inds += inds[:valid_num]
            self.remain_inds += inds[valid_num:]

        print(f"Customized dataset (balance:{balanced}), samples:{len(self.valid_inds)}")

    def set_sub_samples_off(self):
        self.use_sub_inds = False
        self.use_remain_inds = True

    def set_sub_samples_on(self):
        self.use_sub_inds = True
        self.use_remain_inds = False

    def set_all_samples_on(self):
        self.use_sub_inds = False
        self.use_remain_inds = False

    def __getitem__(self, idx):
        if self.use_sub_inds:
            idx = self.valid_inds[idx]
        elif self.use_remain_inds:
            idx = self.remain_inds[idx]
        return self.dataset[idx]

    def __len__(self):
        if self.use_sub_inds:
            return len(self.valid_inds)
        elif self.use_remain_inds:
            return len(self.remain_inds)
        return len(self.dataset)


def get_state_func(method="o", sig_alpha=1e5, transform=None, **kwargs):
    # print("use sig_steepness:", sig_alpha)

    if method == 'o':
        state_func = lambda o, f, g_kl: o
    elif method == 'g_kl':
        state_func = lambda o, f, g_kl: g_kl
    elif method == 'o*g_kl':
        state_func = lambda o, f, g_kl: o * g_kl
    elif method == 'sigmoid(o)':
        state_func = lambda o, f, g_kl: sigmoid(o, sig_alpha=sig_alpha)
    elif method == 'sigmoid(g_kl)':
        state_func = lambda o, f, g_kl: sigmoid(g_kl, sig_alpha=sig_alpha)
    elif method == 'sigmoid(o*g_kl)':
        state_func = lambda o, f, g_kl: sigmoid(o * g_kl, sig_alpha=sig_alpha) \
            if g_kl is not None else sigmoid(o, sig_alpha=sig_alpha)

    elif method == 'tanh(o*g_kl)':
        state_func = lambda o, f, g_kl: torch.tanh(o * g_kl * sig_alpha) \
            if g_kl is not None else sigmoid(o * sig_alpha)

    elif method == 'sigmoid(o*g_kl)_f':
        state_func = lambda o, f, g_kl: sigmoid((o * g_kl)[f], sig_alpha=sig_alpha)
    else:
        raise ValueError(f'{method} is not avaliabe')
    return state_func


def acc(p, y):
    with torch.no_grad():
        correct_flags = p.argmax(1).eq(y) if p.size(1) != 1 else p.gt(0).eq(y)
        correct_num = correct_flags.float().sum().item()
        return correct_num, correct_flags.detach()


def compute_acc(network, loader):
    network.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in (loader):
            x, y = x.cuda(), y.cuda()
            p = network.predict(x)
            correct += acc(p, y)[0]
            total += x.shape[0]
    return correct / total


def get_acc_dict(model, names, loaders):
    acc_dict = {}
    for name, loader in zip(names, loaders):
        acc = compute_acc(model, loader)
        acc_dict[name] = acc
    return acc_dict


def save_statistic(dir, statistics, name="training_statistics"):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(os.path.join(dir, name), 'wb') as f:
        pickle.dump(statistics, f)


def load_statistic(dir, name="training_statistics"):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(os.path.join(dir, name), 'rb') as f:
        statistics = pickle.load(f)
    return statistics


def identity(inputs, batch_first=True, pooling_method="avg"):
    """Return the inputs."""
    return inputs


def identities(*inputs, batch_first=True, pooling_method="avg"):
    """Return all inputs as a tuple."""
    return inputs


def pooling_vit_mlp(hiddens, batch_first=True, pooling_method="avg"):
    """For ViT, we only consider neuron activation states on [CLS] token
    Args:
        hiddens (torch.Tensor): The hidden activations. Should have shape
            (batch_size, n_patches, n_units).
    Returns:
        torch.Tensor: Spatially arranged activations, with shape
            (batch_size, n_units, sqrt(n_patches - 1), sqrt(n_patches - 1)).
    """
    if batch_first:
        batch_size, n_patches, n_units = hiddens.shape
    else:
        #  clip-vit has reversed positions
        n_patches, batch_size, n_units = hiddens.shape
        hiddens = hiddens.permute(1, 0, 2)

    if pooling_method == "avg":
        hiddens = hiddens.sum(dim=1)
    elif pooling_method == "first":
        # We only retain CLS token for the layer-norm of vit since it will be used for classification
        hiddens = hiddens[:, 0]
    elif pooling_method == "max":
        hiddens = torch.amax(hiddens, dim=1)  # max pooling
    return hiddens


def pooling_swin_mlp(hiddens, batch_first=True, pooling_method="avg"):
    """For ViT, we only consider neuron activation states on [CLS] token
    Args:
        hiddens (torch.Tensor): The hidden activations. Should have shape
            (batch_size, n_patches, n_patches,n_units).
    """
    if batch_first:
        batch_size, n_patches, n_patches, n_units = hiddens.shape

    hiddens = hiddens.squeeze(-1).squeeze(-1)
    if len(hiddens.shape) > 2:
        if pooling_method == "avg":
            hiddens = hiddens.sum(dim=[1, 2])  # B, W, H, C
    return hiddens


def avg_pooling(hiddens, batch_first=True, pooling_method="avg"):
    # we directly sum each element, since the denominator can be scaled by sigmoid smooth
    # print(hiddens.shape)
    assert pooling_method == "avg"

    # print(hiddens.shape)
    if pooling_method == "avg":
        return hiddens.sum(dim=[2, 3])  # B, C, W, H
    elif pooling_method == "max":
        return torch.amax(hiddens, dim=[2, 3])  # B, C, W, H


class StatePooling:
    def __init__(self, model):
        self.model = model
        self.batch_first = True
        if 'vit' in model.lower():
            self.pooling = pooling_vit_mlp
        elif 'swin' in model.lower():
            # Swin Transformer employs average pooling for classification
            self.pooling = pooling_swin_mlp
        elif ('resnet' in model.lower()) or \
                ('bit' in model.lower()) or ('mobilenet' in model.lower()):
            self.pooling = avg_pooling
        else:
            self.pooling = identity

        if 'clip' in model.lower():
            self.batch_first = False

    def __call__(self, states, pooling_method="avg"):
        return self.pooling(states, self.batch_first, pooling_method)
