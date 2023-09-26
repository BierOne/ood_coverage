import collections
import os, sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights
from torchvision import transforms as trn
from torch.hub import load_state_dict_from_url

from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.csi_net import CSINet
from openood.networks.udg_net import UDGNet
from openood.networks.cider_net import CIDERNet
from openood.networks.npos_net import NPOSNet
from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor

from openood.postprocessors.nac import nethook
from openood.postprocessors.nac.coverage import make_layer_size_dict, KMNC
from openood.postprocessors.nac.utils import StatePooling, acc, get_state_func, TrainSubset, save_statistic
from openood.postprocessors.nac.instr_state import get_intr_name, kl_grad

# import h5py

import random


def set_seed(seed=-1):
    # Choosing and saving a random seed for reproducibility
    if seed == -1:
        seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
    random.seed(args.seed)  # python random generator
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def get_states_from_intruments(model, data_loader, layer_names,
                               spatial_func=None, unpack=None, pooling_first=False):
    model.eval()
    if spatial_func is None:
        spatial_func = lambda s, m: s
    transform = lambda s, m: spatial_func(s, m).detach() if len(s.shape) > 2 else s.detach()
    if not isinstance(layer_names, list):
        layer_names = [layer_names]
    # list.append may cause memory leakage
    layer_output = {ln: ([], [], []) for ln in layer_names}
    total, correct = 0, 0
    layer_output.update({"labels": [], "preds": []})
    with nethook.InstrumentedModel(model) as instr:
        instr.retain_layers(layer_names, detach=False)
        for i, data in enumerate(tqdm(data_loader)):
            x, y = unpack(data)
            p = model(x)
            correct_num, correct_flags = acc(p, y)
            correct += correct_num
            total += x.shape[0]
            for j, ln in enumerate(layer_names):
                # avg-pool for last layer except the vit.encoder.ln (which employs the first clss token)
                pooling_method = "avg" if ln != "encoder.ln" else "first"
                retain_graph = False if j == len(layer_names) - 1 else True
                b_state = instr.retained_layer(ln)
                b_kl_grad = kl_grad(b_state, p, retain_graph=retain_graph)
                if pooling_first:
                    layer_output[ln][0].append(transform(b_state * b_kl_grad, pooling_method).detach().cpu())
                    layer_output[ln][1].append(correct_flags.detach().cpu())
                else:
                    b_state, b_kl_grad = transform(b_state, pooling_method), transform(b_kl_grad, pooling_method)
                    layer_output[ln][0].append(b_state.detach().cpu())
                    layer_output[ln][1].append(correct_flags.detach().cpu())
                    layer_output[ln][2].append(b_kl_grad.detach().cpu())

            layer_output['labels'].append(y.cpu())
            layer_output['preds'].append(p.detach().argmax(1).cpu())

        final_out = {}
        for ln in layer_names:
            if pooling_first:
                final_out[ln.split("module.")[-1]] = (torch.cat(layer_output[ln][0]),
                                                      torch.cat(layer_output[ln][1]))
            else:
                final_out[ln.split("module.")[-1]] = (torch.cat(layer_output[ln][0]),
                                                      torch.cat(layer_output[ln][1]),
                                                      torch.cat(layer_output[ln][2]))
        final_out['labels'] = torch.cat(layer_output['labels']).numpy().astype(int)
        final_out['preds'] = torch.cat(layer_output['preds']).numpy().astype(int)

        return final_out, correct / total


def main(args):
    if args.id_data == "imagenet200":
        root = "./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default"
    elif args.id_data == "cifar10":
        root = "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
    elif args.id_data == "cifar100":
        root = "./results/cifar100_resnet18_32x32_base_e100_lr0.1_default"

    NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
    MODEL = {
        'cifar10': ResNet18_32x32,
        'cifar100': ResNet18_32x32,
        'imagenet200': ResNet18_224x224,
    }
    try:
        num_classes = NUM_CLASSES[args.id_data]
        model_arch = MODEL[args.id_data]
    except KeyError:
        raise NotImplementedError(f'ID dataset {args.id_data} is not supported.')

    if len(glob(os.path.join(root, 's*'))) == 0:
        raise ValueError(f'No subfolders found in {root}')

    postprocessor_name = args.postprocessor
    for i, subfolder in enumerate(sorted(glob(os.path.join(root, 's*')))):

        # load the pretrained model provided by the user
        if postprocessor_name == 'conf_branch':
            net = ConfBranchNet(backbone=model_arch(num_classes=num_classes),
                                num_classes=num_classes)
        elif postprocessor_name == 'godin':
            backbone = model_arch(num_classes=num_classes)
            net = GodinNet(backbone=backbone,
                           feature_size=backbone.feature_size,
                           num_classes=num_classes)
        elif postprocessor_name == 'rotpred':
            net = RotNet(backbone=model_arch(num_classes=num_classes),
                         num_classes=num_classes)
        elif 'udg' in root:
            backbone = model_arch(num_classes=num_classes)
            net = UDGNet(backbone=backbone,
                         num_classes=num_classes,
                         num_clusters=1000)
        elif 'csi' in root:
            backbone = model_arch(num_classes=num_classes)
            net = CSINet(backbone=backbone,
                         feature_size=backbone.feature_size,
                         num_classes=num_classes)
        elif 'cider' in root:
            backbone = model_arch(num_classes=num_classes)
            net = CIDERNet(backbone,
                           head='mlp',
                           feat_dim=128,
                           num_classes=num_classes)
        elif 'npos' in root:
            backbone = model_arch(num_classes=num_classes)
            net = NPOSNet(backbone,
                          head='mlp',
                          feat_dim=128,
                          num_classes=num_classes)
        else:
            net = model_arch(num_classes=num_classes)

        net.load_state_dict(
            torch.load(os.path.join(subfolder, 'best.ckpt'), map_location='cpu'))
        net.cuda()
        net.eval()

        cache_dir = os.path.join("./cache", args.id_data, args.arch, f"s{i}")
        print(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        _, instr_layers = get_intr_name(args.layer_names, args.arch, net)
        spatial_func = StatePooling(args.arch)
        unpack = lambda b: (b['data'].cuda(), b['label'].cuda())

        print(instr_layers)
        loader_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
        }

        preprocessor = get_default_preprocessor(args.id_data)

        dataloader_dict = get_id_ood_dataloader(args.id_data, "./data",
                                                preprocessor, fsood=False,
                                                **loader_kwargs)

        # id
        print(f'Performing inference on {args.id_data} main-train set (num: {args.valid_num})...', flush=True)

        if args.id_data == "cifar10":
            balanced = False if args.valid_num < 10 else True
        elif args.id_data == "cifar100":
            balanced = False if args.valid_num < 100 else True
        dataset = TrainSubset(dataloader_dict['id']['main_train'].dataset,
                              valid_num=args.valid_num,
                              balanced=balanced)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, drop_last=False)

        layer_output, _ = get_states_from_intruments(net, loader, instr_layers, spatial_func, unpack)
        save_statistic(cache_dir, layer_output, f"{args.id_data}_id_main_train")

        print(f'Performing inference on {args.id_data} sub-train set...', flush=True)
        layer_output, _ = get_states_from_intruments(net, dataloader_dict['id']['sub_train'],
                                                     instr_layers, spatial_func, unpack)
        save_statistic(cache_dir, layer_output, f"{args.id_data}_id_sub_train")

        print(f'Performing inference on {args.id_data} val set...', flush=True)
        layer_output, _ = get_states_from_intruments(net, dataloader_dict['id']['val'], instr_layers, spatial_func, unpack)
        save_statistic(cache_dir, layer_output, f"{args.id_data}_id_val")

        print(f'Performing inference on {args.id_data} test set...', flush=True)
        layer_output, _ = get_states_from_intruments(net, dataloader_dict['id']['test'], instr_layers, spatial_func, unpack)
        save_statistic(cache_dir, layer_output, f"{args.id_data}_id_test")

        # ood
        print(f'Performing inference on {args.id_data} ood val set...', flush=True)
        layer_output, _ = get_states_from_intruments(net, dataloader_dict['ood']['val'], instr_layers, spatial_func, unpack)
        save_statistic(cache_dir, layer_output, f"{args.id_data}_ood_val")

        for dataset_name, ood_dl in dataloader_dict['ood']['near'].items():
            print(f'Performing inference on {dataset_name} near ood set...', flush=True)
            layer_output, _ = get_states_from_intruments(net, ood_dl, instr_layers, spatial_func, unpack)
            save_statistic(cache_dir, layer_output, f"{args.id_data}_ood_near_{dataset_name}")

        for dataset_name, ood_dl in dataloader_dict['ood']['far'].items():
            print(f'Performing inference on {dataset_name} far ood set...', flush=True)
            layer_output, _ = get_states_from_intruments(net, ood_dl, instr_layers, spatial_func, unpack)
            save_statistic(cache_dir, layer_output, f"{args.id_data}_ood_far_{dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',
                        default='resnet18',
                        choices=['resnet18'])
    parser.add_argument(
        '--id-data',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100', 'aircraft', 'cub', 'imagenet200'])
    parser.add_argument('--postprocessor', default='nac')
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--valid-num', type=int, default=1000, help="training data samples for building NAC")
    parser.add_argument('--layer-names', default=['avgpool'], nargs="*", type=str, help="intr layers")
    args = parser.parse_args()
    main(args)
