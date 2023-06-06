"""
Create Time: 29/3/2023
Author: BierOne (lyibing112@gmail.com)
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Optional, Sequence, Tuple, Dict

from deps.dissect.netdissect import nethook
from nac.utils import acc, get_state_func
from nac.instr_state import kl_grad
from tqdm import tqdm


def make_layer_size_dict(model, layer_names, input_shape=(1, 3, 224, 224), pooling_func=None):
    if pooling_func is None:
        pooling_func = lambda s: s
    transform = lambda s: pooling_func(s) if len(s.shape) > 2 else s  # avg-pool for last layer
    input = torch.zeros(*input_shape).cuda()
    layer_size_dict = {}
    # print(model)
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

    def assess(self, model, data_loader, pooling_func=None,
               method=None, save_state=False, **kwargs):
        if method is not None:
            self.method = method
        if kwargs:
            self.method_kwargs = kwargs
        model.to(self.device)
        model.eval()

        print(method)
        for k, v in kwargs.items():
            print(k, v)

        if pooling_func is None:
            pooling_func = lambda s: s
        transform = lambda s: pooling_func(s) if len(s.shape) > 2 else s  # avg-pool for last layer
        state_func = get_state_func(self.method, **self.method_kwargs)

        layer_output = {n: ([], [], []) for n in self.layer_names}
        total, correct = 0, 0
        with nethook.InstrumentedModel(model) as instr:
            instr.retain_layers(self.layer_names, detach=False)
            for i, data in enumerate(tqdm(data_loader)):
                x, y = self.unpack(data, self.device)
                p = model(x)
                correct_num, correct_flags = acc(p, y)
                correct += correct_num
                total += x.shape[0]
                b_layer_state = {}
                for ln in (self.layer_names):
                    b_state = instr.retained_layer(ln)
                    b_kl_grad = kl_grad(b_state, p, retain_graph=False)
                    # if i == 0:
                    #     print(b_kl_grad.shape, b_kl_grad.mean([2,3]).norm(dim=1,)[:3])
                    b_state, b_kl_grad = transform(b_state).detach(), \
                        transform(b_kl_grad).detach()
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
        for b_state, correct_flags, b_kl_grad in tqdm(data_loader):
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


class NAC(Coverage):
    def init_variable(self, hyper: Optional[Dict] = None):
        # For simplicity, we utilize hyper-param O (denoting O^* mentioned in the paper appendix)
        assert 'M' in hyper, 'KMNC has hyper-parameter M and O_star'
        self.estimator_dict = {}
        self.current = 0
        self.M = hyper['M']  # number of intervals
        self.O_star = hyper['O_star']  # minimum activated times to fill an interval
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = Estimator(layer_size, self.M, self.O_star, self.device)

    def add(self, other):
        # check if "other" is a KMNC object
        assert (self.M == other.M) and (self.layer_names == other.layer_names)
        for ln in self.layer_names:
            # print("add", ln)
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

    def assess_uncertainty(self, model, data_loader, pooling_func=None, **kwargs):
        model.to(self.device)
        model.eval()
        if pooling_func is None:
            pooling_func = lambda s: s
        transform = lambda s: pooling_func(s) if len(s.shape) > 2 else s  # avg-pool for last layer
        state_func = get_state_func(self.method, **self.method_kwargs)

        scores = []
        total, correct = 0, 0
        with nethook.InstrumentedModel(model) as instr:
            instr.retain_layers(self.layer_names, detach=False)
            for i, data in enumerate(tqdm(data_loader)):
                x, y = self.unpack(data, self.device)
                b_size = x.shape[0]
                p = model(x)
                correct_num, _ = acc(p, y)
                correct += correct_num
                total += x.shape[0]
                correct_flags = torch.ones_like(y).bool() # we calculate cov for all samples
                b_layer_state = {}
                for ln in (self.layer_names):
                    b_state = instr.retained_layer(ln)
                    b_kl_grad = kl_grad(b_state, p, retain_graph=False)
                    # if i == 0:
                    #     print(b_kl_grad.shape, b_kl_grad.mean([2,3]).norm(dim=1,)[:3])
                    b_state, b_kl_grad = transform(b_state).detach(), \
                        transform(b_kl_grad).detach()
                    out = state_func(b_state, correct_flags, b_kl_grad)
                    b_layer_state[ln] = out
                scores.append(self.single_ood_test(b_layer_state, b_size, **kwargs))
        scores = torch.cat(scores).numpy()
        print("ACC: ", correct/total)
        return scores


    def save(self, path, prefix="cov"):
        os.makedirs(path, exist_ok=True)
        for k, v in self.method_kwargs.items():
            prefix += f"_{k}_{v}"
        name = prefix + f"_states_M{self.M}_{self.method}.pkl"
        print('Saving recorded coverage states in %s/%s...' % (path, name))
        state = {
            'layer_size_dict': self.layer_size_dict,
            'hyper': self.hyper,
            'es_states': {ln: self.estimator_dict[ln].states for ln in self.layer_names},
            'method': self.method,
            'method_kwargs': self.method_kwargs
        }
        torch.save(state, os.path.join(path, name))

    @staticmethod
    def load(path, name, device=None, O_star=None, unpack=None, verbose=True):
        print('Loading saved coverage states in %s/%s...' % (path, name))
        state = torch.load(os.path.join(path, name))
        if O_star is not None and O_star > 0:
            print("original O_star state: ", state['hyper']['O_star'])
            state['hyper']['O_star'] = O_star
        Coverage = NAC(state['layer_size_dict'], device=device,
                        hyper=state['hyper'], unpack=unpack)
        if verbose:
            for k, v in state['hyper'].items():
                print("load hyper params of NAC:", k, v)

        for k, v in state['es_states'].items():
            Coverage.estimator_dict[k].load(v)
        try:
            Coverage.method = state['method']
            Coverage.method_kwargs = state['method_kwargs']
        except:
            print("failed to load method and method_kwargs")
            pass
        return Coverage

class Estimator(object):
    def __init__(self, neuron_num, M=1000, O_star=1, device=None):
        assert O_star > 0, 'minumum activated times O_star should > (or =) 1'
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.M, self.O_star, self.n = M, O_star, neuron_num
        self.thresh = torch.linspace(0., 1., M).view(M, -1).repeat(1, neuron_num).to(self.device)
        self.t_act = torch.zeros(M - 1, neuron_num).to(self.device)  # current activated times under each interval
        self.n_coverage = None

    def add(self, other):
        # check if "other" is an Estimator object
        assert (self.M == other.M) and (self.n == other.n)
        self.t_act += other.t_act

    def update(self, states):
        # thresh -> [M, neuron_num] -> [1, M, neuron_num] ->compare-> [num_data, M, neuron_num]
        # states -> [num_data, neuron_num] -> [num_data, 1, neuron_num] ->compare-> [num_data, M, neuron_num]
        """
        Here is the example to check this code:
            M = 10
            states = torch.rand(2, 8)
            thresh = torch.linspace(0., 1., M).view(M, -1).repeat(1, 8)
            b_act = (states.unsqueeze(1) >= thresh[:M - 1].unsqueeze(0)) & \
                            (states.unsqueeze(1) < thresh[1:M].unsqueeze(0))

            b_act.sum(dim=1)
        """
        b_act = (states.unsqueeze(1) >= self.thresh[:self.M - 1].unsqueeze(0)) & \
                (states.unsqueeze(1) < self.thresh[1:self.M].unsqueeze(0))
        b_act = b_act.sum(dim=0)  # [M, neuron_num]
        self.t_act += b_act  # current activated times under each interval

    def get_score(self, method="norm2"):
        nac_score = torch.min(self.t_act / self.O_star, torch.ones_like(self.t_act))  # [M, neuron_num]
        self.nac_score = nac_score  # [M, neuron_num]
        nac_me = (nac_score.sum(dim=0)) / self.M  # [neuron_num]
        if method == "norm2":
            nac_me = nac_me.norm(p=1).cpu()
        elif method == "avg":
            nac_me = nac_me.mean().cpu()

        t_cov = nac_score.mean(dim=1).cpu().numpy() # detailed coverage score at each interval (for visualization)
        return np.append(t_cov, 0), nac_me

    def ood_test(self, states, method="avg"):
        # nac_score -> [M, neuron_num] -> [1, M, neuron_num] ->compare-> [num_data, M, neuron_num]
        # states -> [num_data, neuron_num] -> [num_data, 1, neuron_num] ->compare-> [num_data, M, neuron_num]
        b_act = (states.unsqueeze(1) >= self.thresh[:self.M - 1].unsqueeze(0)) & \
                (states.unsqueeze(1) < self.thresh[1:self.M].unsqueeze(0))
        scores = (b_act * self.nac_score.unsqueeze(0)).sum(dim=1)  # [num_data, num_n]
        if method == "avg":
            scores = scores.mean(dim=1)
        elif method == "norm2":
            scores = scores.norm(p=1, dim=1)
        # print(scores.shape)
        return scores

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
        self.t_act = torch.zeros(self.M - 1, self.n).to(self.device)  # current activated times under each interval




