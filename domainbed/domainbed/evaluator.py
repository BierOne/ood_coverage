import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader

from benchmark_notes.instr_state import get_states_from_intruments
from benchmark_notes.utils import unpack_data
from benchmark_notes.coverage import KMNC

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



def accuracy_from_loader(algorithm, loader, weights, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc = correct / total
    loss = losssum / total
    return acc, loss


def build_loader(loader_kwargs, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader_kwargs.update(kwargs)
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return loader

def accuracy(algorithm, loader_kwargs, weights, **kwargs):
    loader = build_loader(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, **kwargs)

def neuron_states(algorithm, loader_kwargs, spatialize_feat, layer_names=""):
    loader = build_loader(loader_kwargs)
    return get_states_from_intruments(algorithm, loader, layer_names, spatialize_feat)

class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None,
            get_coverage=False, layer_size_dict=None, coverage_hyper=None, device=None
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        # for coverage
        self.get_coverage = get_coverage
        self.Coverage = KMNC(layer_size_dict, device, coverage_hyper['build_kwargs'], unpack=unpack_data)
        self.coverage_hyper = coverage_hyper

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate_coverage(self, algorithm, use_train=True, step=0, save_dir=None, **kwargs):
        n_test_envs = len(self.test_envs)
        # assert n_test_envs == 1
        states_dict = {}
        accuracies = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # name: env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])
            expected = "in" if use_train else "out"

            valid = ((inout == expected) and (env_num not in self.test_envs))
            if not valid:
                continue

            loader = build_loader(loader_kwargs)
            states, acc, *_ = self.Coverage.assess(algorithm, loader, **kwargs)
            self.Coverage.update(**self.coverage_hyper['update_kwargs'])

            accuracies[name + '_acc'] = acc
            states_dict[name] = states

        method_coverage = {self.coverage_hyper['report_name']: self.Coverage.score()}
        # if save_dir is not None:
        #     self.Coverage.save(save_dir, prefix=f"step{step}")

        self.Coverage.clear()
        return method_coverage, states_dict, accuracies


    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        # assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # name: env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = (self.get_coverage or (self.evalmode == "fast")) and \
                        ((inout == "in") and (env_num not in self.test_envs))
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss = accuracy(algorithm, loader_kwargs, weights, debug=self.debug)
            accuracies[name+'_acc'] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
