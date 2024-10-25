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


def transform_hidden(state, pooling=True, normalize=None):
    d_size, channels, *_ = state.shape  # [dataset_len, neuron_num, W, H] or [dataset_len, neuron_num]
    if len(state.shape) > 2:
        # default: avg pooling
        n_states = state.permute(1, 0, 2, 3).mean(dim=[2, 3]) if pooling \
            else state.permute(1, 0, 2, 3).reshape(channels, -1)
    else:
        n_states = state.permute(1, 0)

    if normalize is not None:
        assert normalize in [0, 1]
        # norm at neuron distribution (0) or feature distribution (1)
        dim = abs(normalize - 1)  # because of the permutation
        n_states = F.normalize(n_states, dim=dim)
    return n_states


def get_layer_state(states_dict, env=None, layers=None):
    if env is None:  # load all envs
        d_names = list(states_dict.keys())
    else:
        d_names = env if isinstance(env, list) else [env]

    if layers is None:  # load all layers
        layers = list(states_dict[d_names[0]].keys())
    elif not isinstance(layers, list):
        layers = [layers]

    layer_dict, acc_dict, kl_grad_dict = [{l: [] for l in layers} for _ in range(3)]
    for d in d_names:
        d_out = env_key_to_dict_data(d, states_dict)
        # print(list(d_out.keys()))
        for l in layers:
            state, correct_flags, grad_kl, *_ = d_out[l]
            layer_dict[l].append(state)
            acc_dict[l].append(correct_flags)
            kl_grad_dict[l].append(grad_kl)
    layer_dict = {l: torch.cat(layer_dict[l]) for l in layer_dict}
    acc_dict = {l: torch.cat(acc_dict[l]) for l in acc_dict}
    kl_grad_dict = {l: torch.cat(kl_grad_dict[l]) for l in kl_grad_dict}
    return layer_dict, acc_dict, kl_grad_dict


def get_single_neuron_state(states_dict, layer, n_id, **kwargs):
    n_states = get_neuron_state(states_dict, layer, **kwargs)
    return n_states[n_id]


def sigmoid(x, smooth=1.0):
    """
    m is the smooth factor (larger denotes smoother)
    """
    return 1 / (1 + torch.exp(-smooth * x))


def get_state_func(method="o", sig_alpha=1e0):
    # print('using sig_smooth', sig_alpha)
    if method == 'o':
        state_func = lambda o, f, g_kl: o
    elif method == 'sigmoid(o)':
        state_func = lambda o, f, g_kl: sigmoid(o, smooth=sig_alpha)
    elif method == 'sigmoid(o*g_kl)':
        state_func = lambda o, f, g_kl: sigmoid(o * g_kl, smooth=sig_alpha)
    elif method == 'sigmoid(o*g_kl[f])':
        state_func = lambda o, f, g_kl: sigmoid((o * g_kl)[f], smooth=sig_alpha)
    else:
        raise ValueError(f'{method} is not avaliabe')

    return state_func


def get_neuron_state(states_dict, layer, env=None, method="o", sig_alpha=1e1,
                     pooling=True, normalize=None, return_func=False, perm=True):
    state_func = get_state_func(method, sig_alpha)
    layer_dict = get_layer_state(states_dict, env=env, layers=layer)
    n_states = transform_hidden(state_func(*[state[layer] for state in layer_dict]),
                                pooling=pooling, normalize=normalize)
    if not perm:
        n_states = n_states.permute(1, 0)  # permute back

    if return_func:
        return state_func, n_states
    else:
        return n_states


def get_mean_minmax(states, dim=1):
    mean = states.mean(dim=dim)
    max, min = states.max(dim=dim)[0], states.min(dim=dim)[0]
    return mean, max, min




def acc(p, y):
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


def get_step_val(dataset_name):
    # step_val = 300 if DATASET_NAME != "DomainNet" else 1000
    if dataset_name == "DomainNet":
        step_val = 1000
    elif "MNIST" in dataset_name:
        step_val = 300
    else:
        step_val = 300
    return step_val


def get_env_name(env_aka, idx_to_env):
    env_id = int(env_aka.split('_')[0][-1])
    env_name = idx_to_env[env_id]
    return env_name


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


def get_best_point(method_coverage, training_statistics, cond="val", te_env=None):
    if cond == "val":
        _, (idx, score) = average_acc_from_dict(training_statistics, key='val')
    elif cond == "test_out":
        _, (idx, score) = average_acc_from_dict(training_statistics, key='test_out')
    elif cond == "test":
        _, (idx, score) = average_acc_from_dict(training_statistics, key='test')
    else:
        auc_tc = [method_coverage[k][cond][2] for k in method_coverage]
        idx, score = np.argmax(auc_tc), np.max(auc_tc)

    cov_dict = list(method_coverage.values())[idx]
    if cond in cov_dict:
        threshs, coverage_list, auctc = cov_dict[cond]
    else:
        threshs, coverage_list, auctc = cov_dict['micro_avg']

    if te_env is not None:
        # get test out acc
        test_acc = list(training_statistics.values())[idx][2][te_env]
    else:
        test_acc = list(training_statistics.values())[idx][-1]

    return test_acc, auctc, score, idx


def average_acc_from_dict(statistics, key='val', cond_func=lambda k: 1):
    if key == 'val':
        idx = 1
    elif key == 'test_out':
        idx = 2
    elif key == 'train':
        idx = 0  # index in the training_statistics[key]
    elif key == 'test':
        idx = 3

    acc_list = []
    for k in statistics:
        if cond_func(k):
            accs = statistics[k][idx]
            if isinstance(accs, dict):
                accs = np.mean([accs[env] for env in accs])
            acc_list.append(accs)
    best_point = [np.argmax(acc_list), np.max(acc_list)]
    return acc_list, best_point


def env_key_to_dict_data(env_key, data_dict):
    if env_key in data_dict:
        v = data_dict[env_key]
    else:
        try:
            env_key = env_key + "_acc"
            v = data_dict[env_key]
        except:
            env_key = env_key.split("_acc")[0]
            v = data_dict[env_key]
    return v


def env_acc_from_dict(statistics, env_key, key='val', cond_func=lambda k: 1):
    if key == 'val':
        env_key = env_key.replace("in", "out")  # in case the env_key is not totally correct
        idx = 1
    elif key == 'test_out':
        env_key = env_key.replace("in", "out")
        idx = 2
    elif key == 'train':
        idx = 0  # index in the training_statistics[key]
    elif key == 'test':
        idx = 3

    acc_list = []
    for k in statistics:
        if cond_func(k):
            accs = statistics[k][idx]
            if isinstance(accs, dict):
                acc_list.append(env_key_to_dict_data(env_key, accs))
            else:
                # we directly load test score
                acc_list.append(accs)
    best_point = [np.argmax(acc_list), np.max(acc_list)]
    return acc_list, best_point


def load_cache(dir, grad=False):
    suffix = '' if not grad else '_grad'
    method_coverage = load_statistic(dir, name="method_coverage" + suffix)
    training_statistics = load_statistic(dir, name="training_statistics")
    for k in training_statistics:
        (tr_acc_dict, val_acc_dict, out_test_acc_dict, test_acc) = training_statistics[k]
        train_domain_names = list(tr_acc_dict.keys())
        val_domain_names = list(val_acc_dict.keys())
        out_test_domain_names = list(out_test_acc_dict.keys())
        break

    return method_coverage, training_statistics, \
        (train_domain_names, val_domain_names, out_test_domain_names)


def identity(inputs, batch_first=True):
    """Return the inputs."""
    return inputs


def identities(*inputs, batch_first=True):
    """Return all inputs as a tuple."""
    return inputs


def pooling_vit_mlp(hiddens: torch.Tensor, batch_first=True) -> torch.Tensor:
    """Make ViT MLP activations look like convolutional activations.
    Each activation corresponds to an image patch, so we can arrange them
    spatially. This allows us to use all the same dissection tools we
    used for CNNs.
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

    hiddens = hiddens.sum(dim=1)
    return hiddens



def unpack_data(data, device):
    return data["x"].to(device), data["y"].to(device)


def avg_pooling(hiddens, batch_first=True, pooling_method="avg"):
    # we directly sum each element, since the denominator can be scaled by sigmoid smooth
    assert pooling_method == "avg"

    if pooling_method == "avg":
        return hiddens.sum(dim=[2, 3])  # B, C, W, H
    elif pooling_method == "max":
        return torch.amax(hiddens, dim=[2, 3])  # B, C, W, H

class FeatSpatialize:
    def __init__(self, model):
        self.model = model
        self.batch_first = True
        if 'vit' in model.lower():
            self.spatialize = pooling_vit_mlp
        elif 'resnet' in model.lower():
            self.spatialize = avg_pooling
        else:
            self.spatialize = identity

        if 'clip' in model.lower():
            self.batch_first = False

    def __call__(self, feat):
        return self.spatialize(feat, self.batch_first)
