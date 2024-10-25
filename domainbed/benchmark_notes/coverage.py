"""
Create Time: 29/3/2023
Author: BierOne (lyibing112@gmail.com)
"""
import os
import torch
import numpy as np
from typing import Any, Callable, Optional, Sequence, Tuple, Dict

from benchmark_notes.utils import get_state_func, acc
from benchmark_notes import nethook
from benchmark_notes.instr_state import kl_grad


def make_layer_size_dict(model, layer_names, input_shape=(1, 3, 224, 224), spatial_func=None):
    if spatial_func is None:
        spatial_func = lambda s: s
    transform = lambda s: spatial_func(s) if len(s.shape) > 2 else s  # avg-pool for last layer
    input = torch.zeros(*input_shape).cuda()
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
                 method: Optional[Any] = None,
                 **kwargs):
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.layer_size_dict = layer_size_dict
        self.layer_names = list(layer_size_dict.keys())
        self.unpack = unpack
        self.coverage_dict = {ln: 0 for ln in self.layer_names}
        self.hyper = hyper
        self.method = method
        self.method_kwargs = kwargs
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

    def assess(self, model, data_loader, spatial_func=None, method=None, save_state=False, **kwargs):
        if method is not None:
            self.method = method
        if kwargs:
            self.method_kwargs = kwargs
        model.to(self.device)
        model.eval()
        if spatial_func is None:
            spatial_func = lambda s: s
        transform = lambda s: spatial_func(s).detach() if len(s.shape) > 2 else s.detach()  # avg-pool for last layer
        state_func = get_state_func(self.method, **self.method_kwargs)

        layer_output = {n: ([], [], []) for n in self.layer_names}
        total, correct = 0, 0
        with nethook.InstrumentedModel(model) as instr:
            instr.retain_layers(self.layer_names, detach=False)
            for i, data in enumerate(data_loader):
                x, y = self.unpack(data, self.device)
                p = model(x)
                correct_num, correct_flags = acc(p, y)
                correct += correct_num
                total += x.shape[0]
                b_layer_state = {}
                for j, ln in enumerate(self.layer_names):
                    retain_graph = False if j == len(self.layer_names) - 1 else True
                    b_state = instr.retained_layer(ln)
                    b_kl_grad = kl_grad(b_state, p, retain_graph=retain_graph)
                    b_state, b_kl_grad = transform(b_state), transform(b_kl_grad)
                    out = state_func(b_state, correct_flags, b_kl_grad)
                    b_layer_state[ln] = out

                    if save_state:
                        layer_output[ln][0].append(b_state.cpu())
                        layer_output[ln][1].append(correct_flags.cpu())
                        layer_output[ln][2].append(b_kl_grad.cpu())
                self.step(b_layer_state)
        if save_state:
            for ln in layer_output:
                states, flags, kl_grads = layer_output[ln]
                layer_output[ln] = (torch.cat(states), torch.cat(flags), torch.cat(kl_grads))
        return layer_output, correct / total

    def assess_with_cache(self, layer_name, data_loader, method=None, **kwargs):
        if method is not None:
            self.method = method
        if kwargs:
            self.method_kwargs = kwargs
        state_func = get_state_func(self.method, **self.method_kwargs)
        b_layer_state = {}
        for b_state, correct_flags, b_kl_grad in (data_loader):
            out = state_func(b_state.to(self.device, non_blocking=True),
                             correct_flags.to(self.device, non_blocking=True),
                             b_kl_grad.to(self.device, non_blocking=True))
            b_layer_state[layer_name] = out
            self.step(b_layer_state)

    def score(self, layer_name=None):
        if len(self.layer_names) == 1:
            layer_name = self.layer_names[0]
        if layer_name:
            return self.coverage_dict[layer_name]
        return self.coverage_dict


class KMNC(Coverage):
    def init_variable(self, hyper: Optional[Dict] = None):
        self.estimator_dict = {}
        self.current = 0

        assert ('M' in hyper and 'O' in hyper)
        self.M = hyper['M']  # number of buckets
        self.O = hyper['O']  # minimum number of samples required for bin filling
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = Estimator(layer_size, self.M, self.O, self.device)

    def add(self, other):
        # check if other is a KMNC object
        assert (self.M == other.M) and (self.layer_names == other.layer_names)
        for ln in self.layer_names:
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

    def save(self, path, prefix="cov"):
        os.makedirs(path, exist_ok=True)
        for k, v in self.method_kwargs.items():
            prefix += f"_{k}_{v}"
        name = prefix + f"_states_M{self.M}_{self.method}.pkl"
        # print('Saving recorded coverage states in %s/%s...' % (path, name))
        state = {
            'layer_size_dict': self.layer_size_dict,
            'hyper': self.hyper,
            'es_states': {ln: self.estimator_dict[ln].states for ln in self.layer_names},
            'method': self.method,
            'method_kwargs': self.method_kwargs
        }
        torch.save(state, os.path.join(path, name))

    @staticmethod
    def load(path, name, device=None, r=None, unpack=None, verbose=False):
        state = torch.load(os.path.join(path, name))
        if r is not None and r > 0:
            state['hyper']['O'] = r
        if verbose:
            print('Loading saved coverage states in %s/%s...' % (path, name))
            for k, v in state['hyper'].items():
                print("load hyper params of Coverage:", k, v)
        Coverage = KMNC(state['layer_size_dict'], device=device,
                        hyper=state['hyper'], unpack=unpack)
        for k, v in state['es_states'].items():
            Coverage.estimator_dict[k].load(v)
        try:
            Coverage.method = state['method']
            Coverage.method_kwargs = state['method_kwargs']
        except:
            print("failed to load method and method_kwargs")
            pass
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
        assert O > 0, 'O should > (or =) 1'
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.M, self.O, self.N = M, O, neuron_num
        self.thresh = torch.linspace(0., 1., M).view(M, -1).repeat(1, neuron_num).to(self.device)
        # self.thresh = logspace(1e3, M).view(M, -1).repeat(1, neuron_num).to(self.device)
        self.t_act = torch.zeros(M - 1, neuron_num).to(self.device)  # current activations under each thresh

    def add(self, other):
        # check if other is an Estimator object
        assert (self.M == other.M) and (self.N == other.N)
        self.t_act += other.t_act

    def update(self, states):
        # bmax, bmin = states.max(dim=0)[0], states.min(dim=0)[0]  # [num_neuron]
        # if (bmax > self.upper).any() or (bmin < self.lower).any():

        # thresh -> [num_t, num_n] -> [1, num_t, num_n] ->compare-> [num_data, num_t, num_n]
        # states -> [num_data, num_n] -> [num_data, 1, num_n] ->compare-> ...
        with torch.no_grad():
            # print(states.shape)
            b_act = (states.unsqueeze(1) >= self.thresh[:self.M - 1].unsqueeze(0)) & \
                    (states.unsqueeze(1) < self.thresh[1:self.M].unsqueeze(0))
            b_act = b_act.sum(dim=0)  # [num_t, num_n]
            self.t_act += b_act  # current activations under each thresh

    def get_score(self, method="avg"):
        t_score = torch.min(self.t_act / self.O, torch.ones_like(self.t_act))  # [num_t, num_n]
        coverage = (t_score.sum(dim=0)) / self.M  # [num_n]
        if method == "norm2":
            coverage = coverage.norm(p=1)
        elif method == "avg":
            coverage = coverage.mean()

        # t_cov = t_score.mean(dim=1).cpu().numpy() # for simplicity
        t_cov = t_score[:, 0].cpu().numpy()  # for simplicity
        return np.append(t_cov, 0), coverage.cpu()

    @property
    def states(self):
        return {
            "thresh": self.thresh.cpu(),
            "t_act": self.t_act.cpu()
        }

    def load(self, state_dict):
        self.thresh = state_dict["thresh"].to(self.device)
        self.t_act = state_dict["t_act"].to(self.device)

    def clear(self):
        self.t_act = torch.zeros(self.M - 1, self.N).to(self.device)  # current activations under each thresh
