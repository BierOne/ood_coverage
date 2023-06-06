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


def get_state_func(method="o", sig_alpha=1e3):
    print("use sig_smooth:", sig_alpha)
    if method == 'o':
        state_func = lambda o, f, g_kl: o
    elif method == 'o[f]':
        state_func = lambda o, f, g_kl: o[f]
    elif method == 'tanh(o)':
        state_func = lambda o, f, g_kl: torch.tanh(o)
    elif method == 'o*g_kl':
        state_func = lambda o, f, g_kl: (o * g_kl)
    elif method == 'sigmoid(o*g_kl)':
        state_func = lambda o, f, g_kl: sigmoid(o * g_kl, sig_alpha=sig_alpha)
    elif method == 'sigmoid(o)':
        state_func = lambda o, f, g_kl: sigmoid(o, sig_alpha=sig_alpha)
    elif method == 'sigmoid(g_kl)':
        state_func = lambda o, f, g_kl: sigmoid(g_kl, sig_alpha=sig_alpha)
    elif method == 'g_kl':
        state_func = lambda o, f, g_kl: g_kl
    elif method == 'abs(g_kl)':
        state_func = lambda o, f, g_kl: g_kl.abs()
    elif method == 'abs(o*g_kl)':
        state_func = lambda o, f, g_kl: (o * g_kl).abs()
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


def identity(inputs, batch_first=True):
    """Return the inputs."""
    return inputs


def identities(*inputs, batch_first=True):
    """Return all inputs as a tuple."""
    return inputs


def pooling_vit_mlp(hiddens: torch.Tensor, batch_first=True) -> torch.Tensor:
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

    # We only retain CLS token since it will be used for feature extraction
    hiddens = hiddens[:, 0]
    return hiddens


class StatePooling:
    def __init__(self, model):
        self.model = model
        self.batch_first = True
        if 'vit' in model.lower():
            self.pooling = pooling_vit_mlp
        elif ('resnet' in model.lower()) or \
            ('bit' in model.lower()) or ('mobilenet' in model.lower()):
            self.pooling = lambda s, b: s.sum(dim=[2, 3]) # because we need the gradient of each element
        else:
            self.pooling = identity

        if 'clip' in model.lower():
            self.batch_first = False

    def __call__(self, states):
        return self.pooling(states, self.batch_first)
