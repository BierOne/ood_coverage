"""
Create Time: 29/3/2023
Author: BierOne (lyibing112@gmail.com)
"""
import os
import torch

import numpy as np
from typing import Any, Callable, Optional, Sequence, Tuple, Dict
from tqdm import tqdm
import time

from . import nethook
from .utils import acc, get_state_func
from .instr_state import kl_grad

from torch.profiler import profile, record_function, ProfilerActivity


def make_layer_size_dict(model, layer_names, input_shape=(3, 3, 224, 224),
                         spatial_func=None, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if spatial_func is None:
        spatial_func = lambda s: s
    transform = lambda s: spatial_func(s) if len(s.shape) > 2 else s  # avg-pool for last layer
    input = torch.zeros(*input_shape).to(device)
    layer_size_dict = {}
    with nethook.InstrumentedModel(model) as instr:
        instr.retain_layers(layer_names, detach=True)
        with torch.no_grad():
            _ = model(input)
            for ln in layer_names:
                b_state = instr.retained_layer(ln)
                layer_size_dict[ln] = transform(b_state).shape[1]
    return layer_size_dict


class Coverage:
    def __init__(self,
                 layer_size_dict: Dict,
                 device: Optional[Any] = None,
                 hyper: Optional[Dict] = None,
                 unpack: Optional[Callable] = None,
                 **kwargs):
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.layer_size_dict = layer_size_dict
        self.layer_names = list(layer_size_dict.keys())
        self.unpack = unpack
        self.coverage_dict = {ln: 0 for ln in self.layer_names}
        self.init_variable(hyper)

    def init_variable(self, hyper):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def step(self, b_layer_state):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path, method):
        raise NotImplementedError

    def assess(self, model, data_loader, **kwargs):
        raise NotImplementedError

    def score(self, layer_name=None):
        if len(self.layer_names) == 1:
            layer_name = self.layer_names[0]
        if layer_name:
            return self.coverage_dict[layer_name]
        return self.coverage_dict


class KMNC(Coverage):
    def init_variable(self, hyper: Optional[Dict] = None):
        self.estimator_dict, hyper_dict = {}, {}
        self.current = 0
        self.hyper_dict = hyper  # number of buckets
        for (ln, layer_size) in self.layer_size_dict.items():
            assert ('M' in hyper[ln] and 'O' in hyper[ln] and
                    'sig_alpha' in hyper[ln] and 'method' in hyper[ln]), \
                'KMNC has hyper-parameter M, O, sig_alpha, method'
            M, O = self.hyper_dict[ln]['M'], self.hyper_dict[ln]['O']
            self.estimator_dict[ln] = Estimator(layer_size, M, O, self.device)

    def add(self, other):
        # check if other is a KMNC object
        for ln in self.layer_names:
            assert (self.hyper_dict['M'] == other.hyper_dict['M']) and \
                   (self.layer_names == other.layer_names)
            self.estimator_dict[ln].add(other.estimator_dict[ln])

    def clear(self):
        for ln in self.layer_names:
            self.estimator_dict[ln].clear()

    def step(self, b_layer_state):
        for (ln, states) in b_layer_state.items():
            if len(states) > 0:
                # print(states.shape)
                self.estimator_dict[ln].update(states)

    def update(self, **kwargs):
        for ln in self.layer_names:
            thresh = self.estimator_dict[ln].thresh[:, 0].cpu().numpy()
            t_cov, coverage = self.estimator_dict[ln].get_score(**kwargs)
            self.coverage_dict[ln] = (thresh, t_cov, coverage)
        return self.score()

    def single_ood_test(self, b_layer_state, b_size, **kwargs):
        layer_num = len(self.layer_names)  # default: layer_num == 1
        scores = torch.zeros(b_size).to(self.device)
        for ln, state in b_layer_state.items():
            scores += self.estimator_dict[ln].ood_test(state, **kwargs) / layer_num
        return scores.cpu()

    def assess(self, model, data_loader, spatial_func=None, **kwargs):
        model.eval()
        if spatial_func is None:
            spatial_func = lambda s, m: s
        transform = lambda s, m: spatial_func(s, m).detach() if len(s.shape) > 2 else s.detach()
        state_funcs = {ln: get_state_func(transform=transform, **self.hyper_dict[ln]) for ln in self.layer_names}

        layer_output = {n: ([], [], []) for n in self.layer_names}
        total, correct = 0, 0

        with nethook.InstrumentedModel(model) as instr:
            instr.retain_layers(self.layer_names, detach=False)
            start_time = time.time()
            for i, data in enumerate(tqdm(data_loader)):
                x, y = self.unpack(data, self.device)
                # data_time = time.time() - start_time
                p = model(x)
                correct_num, correct_flags = acc(p, y)
                correct += correct_num
                total += x.shape[0]
                b_layer_state = {}
                for j, ln in enumerate(self.layer_names):
                    # we use avg-pool for last layer except the vit.encoder.ln (which employs the first clss token)
                    pooling_method = "avg" if ln != "encoder.ln" else "first"
                    retain_graph = False if j == len(self.layer_names) - 1 else True
                    b_state = instr.retained_layer(ln)
                    b_kl_grad = kl_grad(b_state, p, retain_graph=retain_graph)
                    b_state, b_kl_grad = transform(b_state, pooling_method), transform(b_kl_grad, pooling_method)
                    out = state_funcs[ln](b_state, correct_flags, b_kl_grad)
                    b_layer_state[ln] = out
                    # if i == 0:
                    #     print(b_kl_grad.shape, b_kl_grad.mean([2, 3]).norm(dim=1, )[:3])
                self.step(b_layer_state)
                # batch_time = time.time() - start_time
                # if i % 10 == 0:
                # print(f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True)
                # start_time = time.time()
                # break
        return layer_output, correct / total

    def assess_ood(self, model, data_loader, spatial_func=None, progress=True, **kwargs):
        model.eval()
        if spatial_func is None:
            spatial_func = lambda s, m: s
        transform = lambda s, m: spatial_func(s, m).detach() if len(s.shape) > 2 else s.detach()
        state_funcs = {ln: get_state_func(transform=transform, **self.hyper_dict[ln]) for ln in self.layer_names}

        preds, labels, scores, flags = [], [], [], []
        with nethook.InstrumentedModel(model) as instr:
            instr.retain_layers(self.layer_names, detach=False)
            for i, data in enumerate(tqdm(data_loader, disable=not progress)):
                x, y = self.unpack(data, self.device)
                b_size = x.shape[0]
                p = model(x)
                correct_num, b_flags = acc(p, y)
                correct_flags = torch.ones_like(y).bool()  # we calculate cov for all samples
                b_layer_state = {}
                for j, ln in enumerate(self.layer_names):
                    # we use avg-pool for last layer except the vit.encoder.ln (which employs the first clss token)
                    pooling_method = "avg" if ln != "encoder.ln" else "first"
                    retain_graph = False if j == len(self.layer_names) - 1 else True
                    b_state = instr.retained_layer(ln)
                    b_kl_grad = kl_grad(b_state, p, retain_graph=retain_graph)
                    b_state, b_kl_grad = transform(b_state, pooling_method), transform(b_kl_grad, pooling_method)
                    # if i == 0:
                    #     print(b_kl_grad.shape, b_kl_grad.mean([2,3]).norm(dim=1,)[:3])
                    out = state_funcs[ln](b_state, correct_flags, b_kl_grad)
                    b_layer_state[ln] = out
                scores.append(self.single_ood_test(b_layer_state, b_size, **kwargs))
                flags.append(b_flags.cpu())
                preds.append(p.detach().argmax(1).cpu())
                labels.append(y.cpu())
        scores = torch.cat(scores)
        flags = torch.cat(flags)
        preds = torch.cat(preds).numpy().astype(int)
        labels = torch.cat(labels).numpy().astype(int)
        # print("ACC: ", flags.float().mean())
        return scores, flags, preds, labels

    def assess_ood_with_cache(self, loader_dict, progress=True, **kwargs):
        state_funcs = {ln: get_state_func(**self.hyper_dict[ln]) for ln in self.layer_names}
        total_scores, flags = [], []
        for layer_name in self.layer_names:
            scores = []
            for b_state, correct_flags, b_kl_grad in tqdm(loader_dict[layer_name], disable=not progress):
                correct_flags = torch.ones_like(correct_flags).bool()  # we calculate cov for all samples
                b_size = b_state.shape[0]
                out = state_funcs[layer_name](b_state.to(self.device, non_blocking=True),
                                              correct_flags.to(self.device, non_blocking=True),
                                              b_kl_grad.to(self.device, non_blocking=True))
                b_layer_state = {layer_name: out}
                scores.append(self.single_ood_test(b_layer_state, b_size, **kwargs))
            total_scores.append(torch.cat(scores).numpy())
        return np.vstack(total_scores).mean(0), flags, loader_dict['preds'], loader_dict['labels']

    def assess_with_cache(self, loader_dict, **kwargs):
        state_funcs = {ln: get_state_func(**self.hyper_dict[ln]) for ln in self.layer_names}
        for layer_name in self.layer_names:
            for b_state, correct_flags, b_kl_grad in tqdm(loader_dict[layer_name]):
                out = state_funcs[layer_name](b_state.to(self.device, non_blocking=True),
                                              correct_flags.to(self.device, non_blocking=True),
                                              b_kl_grad.to(self.device, non_blocking=True))
                b_layer_state = {layer_name: out}
                self.step(b_layer_state)

    def save(self, path, prefix="imagenet", aka_ln=None, verbose=True):
        if aka_ln is None:
            aka_ln = {ln: ln for ln in self.hyper_dict}
        os.makedirs(path, exist_ok=True)
        for ln, ln_hypers in self.hyper_dict.items():
            ln_prefix = [f"{k}_{v}" for k, v in ln_hypers.items() if k != 'O']
            prefix += f"_{aka_ln[ln]}_" + "_".join(ln_prefix)
        name = prefix + f"_states.pkl"
        if verbose:
            print('Saving recorded coverage states in %s/%s...' % (path, name))
        state = {
            'layer_size_dict': self.layer_size_dict,
            'hyper_dict': self.hyper_dict,
            'es_states': {ln: self.estimator_dict[ln].states for ln in self.layer_names},
        }
        torch.save(state, os.path.join(path, name))

    @staticmethod
    def load(path, name, device=None, params=None, unpack=None, verbose=False):
        state = torch.load(os.path.join(path, name))
        if params is not None:
            # print("Update params:", params)
            state['hyper_dict'].update(params)
        Coverage = KMNC(state['layer_size_dict'], device=device,
                        hyper=state['hyper_dict'], unpack=unpack)
        if verbose:
            print('Loading saved coverage states in %s/%s...' % (path, name))
            for ln, ln_hypers in state['hyper_dict'].items():
                ln_prefix = [f"{k}:{v}, " for k, v in ln_hypers.items()]
                print("load hyper params of Coverage:", ln_prefix)
        for k, v in state['es_states'].items():
            Coverage.estimator_dict[k].load(v)
        return Coverage


def logspace(base=10, num=100):
    num = int(num / 2)
    x = np.linspace(1, np.sqrt(base), num=num)
    x_l = np.emath.logn(base, x)
    x_r = (1 - x_l)[::-1]
    x = np.concatenate([x_l[:-1], x_r])
    x[-1] += 1e-2
    return torch.from_numpy(np.append(x, 1.2))



class Estimator(object):
    def __init__(self, neuron_num, M=1000, O=1, device=None):
        assert O > 0, 'minumum activated number O should > (or =) 1'
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.M, self.O, self.N = M, O, neuron_num
        # self.thresh = torch.linspace(0., 1.01, M).view(M, -1).repeat(1, neuron_num).to(self.device)
        self.thresh = logspace(1e3, M).view(M, -1).repeat(1, neuron_num).to(self.device)
        self.t_act = torch.zeros(M - 1, neuron_num).to(self.device)  # current activations under each thresh
        self.n_coverage = None

    def add(self, other):
        # check if other is an Estimator object
        assert (self.M == other.M) and (self.N == other.N)
        self.t_act += other.t_act

    def update(self, states):
        # thresh -> [num_t, num_n] -> [1, num_t, num_n] ->compare-> [num_data, num_t, num_n]
        # states -> [num_data, num_n] -> [num_data, 1, num_n] ->compare-> ...
        """
        Here is the example to check this code:
            k = 10
            states = torch.rand(2, 8)
            thresh = torch.linspace(0., 1., M).view(M, -1).repeat(1, 8)
            b_act = (states.unsqueeze(1) >= thresh[:M - 1].unsqueeze(0)) & \
                            (states.unsqueeze(1) < thresh[1:M].unsqueeze(0))

            b_act.sum(dim=1)
        """
        with torch.no_grad():
            b_act = (states.unsqueeze(1) >= self.thresh[:self.M - 1].unsqueeze(0)) & \
                    (states.unsqueeze(1) < self.thresh[1:self.M].unsqueeze(0))
            b_act = b_act.sum(dim=0)  # [num_t, num_n]
            # print(states.shape[0], b_act.sum(0)[:3])

            self.t_act += b_act  # current activation times under each interval

    def get_score(self, method="avg"):
        t_score = torch.min(self.t_act / self.O, torch.ones_like(self.t_act))  # [num_t, num_n]
        coverage = (t_score.sum(dim=0)) / self.M  # [num_n]
        if method == "norm2":
            coverage = coverage.norm(p=1).cpu()
        elif method == "avg":
            coverage = coverage.mean().cpu()

        t_cov = t_score.mean(dim=1).cpu().numpy()  # for simplicity
        self.n_coverage = t_score  # [num_t, num_n]
        return np.append(t_cov, 0), coverage

    def ood_test(self, states, method="avg"):
        # thresh -> [num_t, num_n] -> [1, num_t, num_n] ->compare-> [num_data, num_t, num_n]
        # states -> [num_data, num_n] -> [num_data, 1, num_n] ->compare-> ...
        b_act = (states.unsqueeze(1) >= self.thresh[:self.M - 1].unsqueeze(0)) & \
                (states.unsqueeze(1) < self.thresh[1:self.M].unsqueeze(0))
        scores = (b_act * self.n_coverage.unsqueeze(0)).sum(dim=1)  # [num_data, num_n]
        if method == "avg":
            scores = scores.mean(dim=1)
        return scores

    @property
    def states(self):
        return {
            "thresh": self.thresh.cpu(),
            "t_act": self.t_act.cpu()
        }

    def load(self, state_dict, zero_corner=True):
        self.thresh = state_dict["thresh"].to(self.device)
        self.t_act = state_dict["t_act"].to(self.device)

    def clear(self):
        self.t_act = torch.zeros(self.M - 1, self.N).to(self.device)  # current activations under each thresh
