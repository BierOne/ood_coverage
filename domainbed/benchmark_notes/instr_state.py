"""
Create Time: 24/3/2023
Author: BierOne (lyibing112@gmail.com)
"""
import os, sys
import torch
from torch import nn
from collections.abc import Iterable
from torch import cat
from domainbed import algorithms
import torch.nn.functional as F

from benchmark_notes import nethook
from tqdm import tqdm
from easydict import EasyDict as edict
from benchmark_notes.utils import get_acc_dict, get_step_val, acc
from benchmark_notes.utils import load_statistic, FeatSpatialize

from domainbed import datasets
import numpy as np


def get_intr_name(backbone, algorithm, dataset, suffix="norm2"):
    if "MNIST" not in dataset:
        if "clip_vit" in backbone:
            instr_layers = ['featurizer.network.ln_post'] if suffix == "" else \
                ['featurizer.network.transformer.resblocks.11.mlp.gelu']
        elif "vit" in backbone:
            instr_layers = ['featurizer.network.fc_norm'] if suffix == "" else \
                ['featurizer.network.blocks.11.norm2']
        elif "swag" in backbone:
            instr_layers = ['featurizer.network.head']
        else:
            instr_layers = ["featurizer.network.avgpool"]

        if "SMA" in algorithm:
            instr_layers = [l.replace("featurizer", "network_sma.0") for l in instr_layers]
    else:
        instr_layers = ["featurizer.bn3"]
    return instr_layers


def kl_grad(b_state, outputs, temperature=1.0, retain_graph=False, **kwargs):
    """
    This implementation follows https://github.com/deeplearning-wisc/gradnorm_ood
    """
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    num_classes = outputs.shape[-1]
    targets = torch.ones_like(outputs) / num_classes

    loss = (torch.sum(-targets * logsoftmax(outputs), dim=-1))
    layer_grad = torch.autograd.grad(loss.sum(), b_state, create_graph=False,
                                     retain_graph=retain_graph, **kwargs)[0]
    return layer_grad




def get_states_from_intruments(network, loader, layer_names, spatial_func=None, rq=None):
    if spatial_func is None:
        spatial_func = lambda s: s
    transform = lambda s: spatial_func(s) if len(s.shape) > 2 else s  # avg-pool for last layer

    if not isinstance(layer_names, list):
        layer_names = [layer_names]
    # list.append may cause memory leakage
    layer_output = {n: ([], [], []) for n in layer_names}
    network.eval()
    total, correct = 0, 0
    with nethook.InstrumentedModel(network) as instr:
        instr.retain_layers(layer_names, detach=False)

        for i, batch in enumerate(loader):
            x = batch["x"].cuda()
            y = batch["y"].cuda()
            p = network.predict(x)
            correct_num, correct_flags = acc(p, y)
            correct += correct_num
            total += x.shape[0]
            for t, ln in enumerate(layer_names):
                b_state = instr.retained_layer(ln)
                b_kl_grad = kl_grad(b_state, p, retain_graph=True)
                # b_kl_grad = rce(b_state, p, y, retain_graph=True)
                layer_output[ln][0].append(transform(b_state).detach().cpu())
                layer_output[ln][1].append(correct_flags.cpu())
                layer_output[ln][2].append(transform(b_kl_grad).detach().cpu())

        for ln in layer_output:
            states, flags, kl_grads = layer_output[ln]
            layer_output[ln] = (torch.cat(states), torch.cat(flags), torch.cat(kl_grads))
            # print([t.shape for t in layer_output[ln]])
        return layer_output, correct / total


def get_states_dict(model, names, loaders, layers=None, sp_func=None):
    states_dict = {}
    acc_dict = {}
    for name, loader in zip(names, loaders):
        states, acc = get_states_from_intruments(model, loader, layers, sp_func)
        states_dict[name] = states
        acc_dict[name] = acc
    return states_dict, acc_dict


from sconf import Config


def load_acc_logs(logs):
    # train-val model selection
    val_acc_dict = logs.get('out_train_acc_dict', {})
    get_val = False if "out_train_acc_dict" in logs else True
    # oracle model selection
    out_test_acc_dict = logs.get('out_test_acc_dict', {})
    get_out_test = False if "out_test_acc_dict" in logs else True
    # model test score
    test_acc = logs.get('test_acc', 0)
    get_test = False if test_acc > 0 else True
    print("get_val:{}, get_out_test:{}, get_test:{}".format(get_val, get_out_test, get_test))
    return (get_val, get_out_test, get_test), (val_acc_dict, out_test_acc_dict, test_acc)


def compute_statistics(args, prefix, layer_names, dataset_stat=None,
                       step_val=300, min_step=0, max_step=5001, best_name=None):
    method_coverage, training_statistics, state_statistics, tr_acc_dict = {}, {}, {}, {}
    model, dataset, trial_seed = None, None, -1

    fmt = "model_step{}.pkl" if best_name is None else f"{best_name}.pkl"
    key_to_model_path = range(min_step, max_step, step_val) if best_name is None else [""]
    if dataset_stat is not None:
        (train_domain_names, train_domain_loaders), (val_domain_names, val_domain_loaders), \
            (test_domain_names, test_domain_loaders), (out_test_domain_names, out_test_domain_loaders) = dataset_stat
    ################### --- Running Statistics --- ###################
    for key in tqdm(key_to_model_path):
        model_path = os.path.join(args.model_dir, prefix, fmt.format(key))
        logs = torch.load(model_path)
        print('Loaded Model Args from {}:'.format(model_path))
        (get_val, get_out_test, get_test), \
            (val_acc_dict, out_test_acc_dict, test_acc) = load_acc_logs(logs)

        hparams = Config(logs['model_hparams'])
        m_args = edict(logs['args'])
        algorithm_class = algorithms.get_algorithm_class(m_args.algorithm)
        if (dataset is None) or ((get_test + get_out_test + get_val) > 0) or \
                ((dataset.test_envs != m_args.real_test_envs) or
                 (dataset.__class__.__name__ != m_args.dataset)):
            print("load dataset", m_args.dataset, m_args.real_test_envs)
            ################### --- Create Dataset --- ###################
            if trial_seed != m_args.trial_seed:
                trial_seed = m_args.trial_seed
                dataset, (train_domain_names, train_domain_loaders), (val_domain_names, val_domain_loaders), \
                    (test_domain_names, test_domain_loaders), (out_test_domain_names, out_test_domain_loaders) = \
                    get_loaders(m_args, hparams, algorithm_class, get_val=get_val, get_out_test=get_out_test,
                                get_test=get_test or args.save_test, workers=args.workers, batch_size=args.batch_size)

        ############## load model #############
        m_name = prefix + fmt.format(key)[:-4]
        model = get_model(m_args, dataset, hparams, logs, algorithm_class, old_model=model)
        ############## get coverage statistics #############
        if args.save_test:
            get_test = False
            # print(train_domain_loaders, train_domain_names)
            states_dict, te_acc_dict = get_states_dict(model, test_domain_names,
                                                       test_domain_loaders, layer_names,
                                                       sp_func=FeatSpatialize(hparams['model']))
            test_acc = np.mean([te_acc_dict[k] for k in te_acc_dict])
        else:
            states_dict, tr_acc_dict = get_states_dict(model, train_domain_names,
                                                       train_domain_loaders, layer_names,
                                                       sp_func=FeatSpatialize(hparams['model']))
        state_statistics[m_name] = states_dict
        ############## get acc (val | test in & out) statistics #############
        if get_val:
            val_acc_dict = get_acc_dict(model, val_domain_names, val_domain_loaders)
        if get_out_test:
            out_test_acc_dict = get_acc_dict(model, out_test_domain_names, out_test_domain_loaders)
        if get_test:
            te_acc_dict = get_acc_dict(model, test_domain_names, test_domain_loaders)
            test_acc = np.mean([te_acc_dict[k] for k in te_acc_dict])

        print("test_acc:", test_acc)
        training_statistics[m_name] = (tr_acc_dict, val_acc_dict, out_test_acc_dict, test_acc)
    return method_coverage, state_statistics, training_statistics


def load_statistic_from_ckt(args, prefix, get_coverage=True, get_states=True, get_train_states=True,
                            step_val=300, min_step=0, max_step=5001):
    method_coverage, training_statistics, state_statistics = {}, {}, {}
    fmt = "model_step{}.pkl"
    key_to_model_path = range(min_step, max_step, step_val)

    ################### --- Running Statistics --- ###################
    for key in tqdm(key_to_model_path):
        model_path = os.path.join(args.model_dir, prefix, fmt.format(key))
        m_name = prefix + fmt.format(key)[:-4]
        dir = os.path.join(args.model_dir, prefix)

        if get_states and os.path.isfile(os.path.join(dir, f"state_statistics_step{key}.pkl")):
            state_statistics[m_name] = load_statistic(dir, f"state_statistics_step{key}")

        if get_coverage or get_train_states:
            logs = torch.load(model_path)
            print('Loaded Model Args from {}:'.format(model_path))

            assert ("out_train_acc_dict" in logs) and \
                   ("out_test_acc_dict" in logs) and "test_acc" in logs
            test_acc = logs['test_acc']  # if not saved, will be zero
            val_acc_dict = logs['out_train_acc_dict']  # train-val model selection
            out_test_acc_dict = logs['out_test_acc_dict']  # oracle model selection
            coverage_dict = logs['coverage']  # oracle model selection

            method_coverage[m_name] = coverage_dict  # {env: (thresh_list, coverage_list, auc)}

            print("test_acc:", test_acc)
            training_statistics[m_name] = (None, val_acc_dict, out_test_acc_dict, test_acc)

    return method_coverage, state_statistics, training_statistics


from domainbed.datasets import get_dataset, split_dataset
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

def get_model(m_args, dataset, hparams, logs,
              algorithm_class=None, old_model=None):
    if old_model:
        old_model.load_state_dict(logs['model_dict'], strict=False)
        return old_model.cuda().eval()
    algorithm_name = m_args.algorithm
    if m_args.algorithm == "GroupDRO":
        algorithm_name = "ERM"
    if algorithm_class is None:
        algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    model = algorithm_class(dataset.input_shape, dataset.num_classes,
                            len(dataset) - len(m_args.real_test_envs), hparams)
    model.load_state_dict(logs['model_dict'], strict=False)
    return model.cuda().eval()


def get_loaders(m_args, hparams,
                algorithm_class=None,
                get_val=False,
                get_out_test=False,
                get_test=False,
                workers=3,
                batch_size=128):
    (train_domain_names, train_domain_loaders), \
        (val_domain_names, val_domain_loaders), \
        (test_domain_names, test_domain_loaders), \
        (out_test_domain_names, out_test_domain_loaders) = (None, None), (None, None), (None, None), (None, None)

    # in_splits, out_splits = create_in_out_splits(dataset, m_args, hparams)
    dataset, in_splits, out_splits = get_dataset(m_args.real_test_envs, m_args, hparams, algorithm_class)

    ################## training data ##################
    train_domain_loaders = [FastDataLoader(
        dataset=env,
        batch_size=batch_size,
        num_workers=workers)
        for i, (env, _) in enumerate(in_splits) if i not in m_args.real_test_envs]

    train_domain_names = ['env{}_in_acc'.format(i) for i in range(len(in_splits))
                          if i not in m_args.real_test_envs]

    # print(m_args.test_envs, m_args.real_test_envs,
    #       [i for i in range(len(in_splits)) if i in m_args.real_test_envs])

    ################## validation data
    if get_val:
        val_domain_loaders = [FastDataLoader(
            dataset=env,
            batch_size=batch_size,
            num_workers=workers)
            for i, (env, _) in enumerate(out_splits) if i not in m_args.real_test_envs]

        val_domain_names = ['env{}_out_acc'.format(i) for i in range(len(in_splits))
                            if i not in m_args.real_test_envs]

    ################## testing (in) data
    if get_test:
        test_domain_loaders = [FastDataLoader(
            dataset=env,
            batch_size=batch_size,
            num_workers=workers)
            for i, (env, _) in enumerate(in_splits) if i in m_args.real_test_envs]
        test_domain_names = ['env{}_in_acc'.format(i) for i in range(len(in_splits))
                             if i in m_args.real_test_envs]

    ################## testing (out) data
    if get_out_test:
        out_test_domain_loaders = [FastDataLoader(
            dataset=env,
            batch_size=batch_size,
            num_workers=workers)
            for i, (env, _) in enumerate(out_splits) if i in m_args.real_test_envs]
        out_test_domain_names = ['env{}_out_acc'.format(i) for i in range(len(in_splits))
                                 if i in m_args.real_test_envs]

    return dataset, (train_domain_names, train_domain_loaders), (val_domain_names, val_domain_loaders), \
        (test_domain_names, test_domain_loaders), (out_test_domain_names, out_test_domain_loaders),
