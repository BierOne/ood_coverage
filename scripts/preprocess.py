import collections
import os, sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights
from torchvision import transforms as trn
from torch.hub import load_state_dict_from_url

from openood.networks import ResNet50, Swin_T, ViT_B_16
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.cider_net import CIDERNet
from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader

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
    random.seed(seed)  # python random generator
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def get_states_from_intruments(model, data_loader, layer_names,
                               spatial_func=None, unpack=None):
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
                b_state, b_kl_grad = transform(b_state, pooling_method), transform(b_kl_grad, pooling_method)
                layer_output[ln][0].append(b_state.detach().cpu())
                layer_output[ln][1].append(correct_flags.detach().cpu())
                layer_output[ln][2].append(b_kl_grad.detach().cpu())

            layer_output['labels'].append(y.cpu())
            layer_output['preds'].append(p.detach().argmax(1).cpu())

        final_out = {}
        for ln in layer_names:
            # final_out[ln.split("module.")[-1]] = (torch.cat(layer_output[ln][0]),
            #                                       torch.cat(layer_output[ln][1]))
            final_out[ln.split("module.")[-1]] = (torch.cat(layer_output[ln][0]),
                                                  torch.cat(layer_output[ln][1]),
                                                  torch.cat(layer_output[ln][2]))
        final_out['labels'] = torch.cat(layer_output['labels']).numpy().astype(int)
        final_out['preds'] = torch.cat(layer_output['preds']).numpy().astype(int)

        return final_out, correct / total


def main(args):
    # set_seed()
    if args.aug == "augmix":
        args.ckpt_path = "./results/imagenet_resnet50_tvsv1_augmix_default/ckpt.pth"
    elif args.aug == "pixmix":
        args.ckpt_path = "./results/imagenet_resnet50_tvsv1_base_pixmix/ckpt.pth"
    elif args.aug == "regmix":
        args.ckpt_path = "./results/imagenet_resnet50_regmixup_e30_lr0.001_alpha10_default/s0/best.ckpt"
    elif args.aug == "deepaug":
        args.ckpt_path = "./results/imagenet_resnet50_tvsv1_base_deepaugment/ckpt.pth"
    elif args.aug == "style":
        args.ckpt_path = "./results/imagenet_resnet50_tvsv1_base_stylized/ckpt.pth"
    elif args.aug == "randaug":
        args.ckpt_path = "./results/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt"
    elif args.aug == "cider":
        args.ckpt_path = "./results/imagenet_cider_net_cider_e10_lr0.001_protom0.95_default/s0/best.ckpt"
    elif args.aug == "rotpred":
        args.ckpt_path = "./results/imagenet_rot_net_rotpred_e30_lr0.001_default/s0/best.ckpt"
    elif args.aug == "godin":
        args.ckpt_path = "./results/imagenet_godin_net_godin_e30_lr0.001_default/s0/best.ckpt"
    elif args.aug == "conf_branch":
        args.ckpt_path = "./results/imagenet_conf_branch_net_conf_branch_e30_lr0.001_default/s0/best.ckpt"
    elif args.aug == "logitnorm":
        args.ckpt_path = "./results/imagenet_resnet50_logitnorm_e30_lr0.001_alpha0.04_default/s0/best.ckpt"
    else:
        args.ckpt_path = None

    if not args.tvs_pretrained:
        assert args.ckpt_path is not None
        root = '/'.join(args.ckpt_path.split('/')[:-1])
    else:
        root = os.path.join(
            ROOT_DIR, 'results',
            f'imagenet_{args.arch}_tvsv{args.tvs_version}_base_default')
        if not os.path.exists(root):
            os.makedirs(root)

    if args.tvs_pretrained and (args.ckpt_path is None):
        if args.arch == 'resnet50':
            net = ResNet50()
            weights = eval(f'ResNet50_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
            preprocessor = weights.transforms()
        elif args.arch == 'swin-t':
            net = Swin_T()
            weights = eval(f'Swin_T_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
            preprocessor = weights.transforms()
        elif args.arch == 'vit-b-16':
            net = ViT_B_16()
            weights = eval(f'ViT_B_16_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
            preprocessor = weights.transforms()
        else:
            raise NotImplementedError
    else:
        if args.arch == 'resnet50':
            if 'conf_branch' in args.ckpt_path:
                net = ConfBranchNet(backbone=ResNet50(), num_classes=1000)
            elif 'godin' in args.ckpt_path:
                backbone = ResNet50()
                net = GodinNet(backbone=backbone,
                               feature_size=backbone.feature_size,
                               num_classes=1000)
            elif 'rotpred' in args.ckpt_path:
                net = RotNet(backbone=ResNet50(), num_classes=1000)
            elif 'cider' in args.ckpt_path:
                net = CIDERNet(backbone=ResNet50(),
                               head='mlp',
                               feat_dim=128,
                               num_classes=1000)
            else:
                net = ResNet50()
            print("load checkpoint from: ", args.ckpt_path)
            ckpt = torch.load(args.ckpt_path, map_location='cpu')
            net.load_state_dict(ckpt)
            preprocessor = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])
        else:
            raise NotImplementedError

    net.cuda()
    net.eval()

    if args.aug != "cross":
        cache_dir = os.path.join("./cache", args.id_data, args.arch, args.aug, str(args.valid_num))
    else:
        cache_dir = os.path.join("./cache", args.id_data, args.arch, str(args.valid_num))
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
    dataloader_dict = get_id_ood_dataloader(args.id_data, "./data",
                                            preprocessor, fsood=False,
                                            **loader_kwargs)

    # id
    print(f'Performing inference on {args.id_data} main-train set (num: {args.valid_num})...', flush=True)

    balanced = False if args.valid_num < 1000 else True
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
                        default='resnet50',
                        choices=['resnet50', 'swin-t', 'vit-b-16'])
    parser.add_argument(
        '--id-data',
        type=str,
        default='imagenet',
        choices=['imagenet'])
    parser.add_argument('--tvs-version', default=1, choices=[1, 2])
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--aug', default="cross", help="training scheme( default: cross entropy training)",
                        choices=["cross", "augmix", "pixmix", "regmix",
                                 "deepaug", "style", "randaug", "logitnorm",
                                 "cider", "rotpred", "godin", "conf_branch"])
    parser.add_argument('--tvs-pretrained', action='store_true')
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--valid-num', type=int, default=1000, help="training data samples for building NAC")
    parser.add_argument('--layer-names', default=['avgpool'], nargs="*", type=str, help="intr layers")
    args = parser.parse_args()
    main(args)
