import collections
import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights
from torchvision import transforms as trn
from torch.hub import load_state_dict_from_url

from openood.evaluation_api import Evaluator

from openood.networks import ResNet50, Swin_T, ViT_B_16
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.cider_net import CIDERNet


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

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

seed = set_seed()
parser = argparse.ArgumentParser()
parser.add_argument('--arch',
                    default='resnet50',
                    choices=['resnet50', 'swin-t', 'vit-b-16'])
parser.add_argument('--tvs-version', default=1, choices=[1, 2])
parser.add_argument('--ckpt-path', default=None)
parser.add_argument('--aug', default="cross", choices=["cross", "augmix", "pixmix", "regmix",
                                                       "deepaug", "style", "randaug", "logitnorm",
                                                       "cider", "rotpred", "godin", "conf_branch"])

parser.add_argument('--tvs-pretrained', action='store_true')
parser.add_argument('--postprocessor', default='msp')
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--batch-size', default=200, type=int)
parser.add_argument('--aps', action='store_true')
parser.add_argument('--use_cache', action='store_true')

parser.add_argument('--valid-num', type=int, default=10000)
parser.add_argument('--layer-names', default=['avgpool'], nargs="*", type=str, help="intr layers")
args = parser.parse_args()

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

# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor
# load pre-setup postprocessor if exists
if os.path.isfile(
        os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl')):
    with open(
            os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl'),
            'rb') as f:
        postprocessor = pickle.load(f)
else:
    postprocessor = None



# assuming the model is either
# 1) torchvision pre-trained; or
# 2) a specified checkpoint
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
        if ('conf_branch' in args.ckpt_path) or (postprocessor_name == 'conf_branch'):
            net = ConfBranchNet(backbone=ResNet50(), num_classes=1000)
        elif ('godin' in args.ckpt_path) or (postprocessor_name == 'godin'):
            backbone = ResNet50()
            net = GodinNet(backbone=backbone,
                           feature_size=backbone.feature_size,
                           num_classes=1000)
        elif ('rotpred' in args.ckpt_path) or (postprocessor_name == 'rotpred'):
            net = RotNet(backbone=ResNet50(), num_classes=1000)
        elif('cider' in args.ckpt_path) or (postprocessor_name == 'cider'):
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

if postprocessor_name == "nac":
    postpc_kwargs = {"valid_num": args.valid_num,
                     "layer_names": args.layer_names,
                     "aps": args.aps,
                     "log_dir": f"logs/{args.aug}_{args.arch}_imagenet_{args.valid_num}"}
    if args.arch == 'resnet50':
        name = 'resnet'
    elif args.arch == 'swin-t':
        name = 'swin'
    elif args.arch == 'vit-b-16':
        name = 'vit'
    else:
        raise ValueError(args.arch)
    config_root = os.path.join(ROOT_DIR, 'configs/postprocessors', 'nac', name)

else:
    postpc_kwargs = {}
    config_root = os.path.join(ROOT_DIR, 'configs')


if args.aug == 'cross':
    cached_dir = os.path.join("./cache", "imagenet", args.arch, str(args.valid_num))
else:
    cached_dir = os.path.join("./cache", "imagenet", args.arch, args.aug, str(args.valid_num))

print(cached_dir)
# a unified evaluator
evaluator = Evaluator(
    net,
    id_name='imagenet',  # the target ID dataset
    data_root=os.path.join(ROOT_DIR, 'data'),
    config_root=config_root,
    preprocessor=preprocessor,  # default preprocessing
    postprocessor_name=postprocessor_name,
    postprocessor=postprocessor,
    batch_size=args.
    batch_size,  # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=8,
    fsood=args.fsood,
    cached_dir=cached_dir,
    use_cache=args.use_cache,
    **postpc_kwargs
)

# the metrics is a dataframe
metrics = evaluator.eval_ood(fsood=args.fsood)

# saving and recording
if args.save_csv:
    saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)

    if not os.path.isfile(
            os.path.join(saving_root, f'{postprocessor_name}.csv')):
        metrics.to_csv(os.path.join(saving_root, f'{postprocessor_name}.csv'),
                       float_format='{:.2f}'.format)

if args.save_score:
    score_save_root = os.path.join(root, 'scores')
    if not os.path.exists(score_save_root):
        os.makedirs(score_save_root)
    with open(os.path.join(score_save_root, f'{postprocessor_name}.pkl'),
              'wb') as f:
        pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)