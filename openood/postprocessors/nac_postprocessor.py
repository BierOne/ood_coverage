from typing import Any
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from .base_postprocessor import BasePostprocessor

from .nac.utils import StatePooling, TrainSubset
from .nac.instr_state import get_intr_name
from .nac.coverage import make_layer_size_dict, KMNC


class NACPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NACPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.test_method = self.args.test_method
        self.layer_names = self.args.layer_names
        self.valid_num = self.args.valid_num

        self.layer_kwargs = self.config.postprocessor.layer_kwargs
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        self.model_name = None
        self.spatial_func = None
        self.Coverage = None
        self.build_nac_flag = True if not self.config.postprocessor.APS_mode else False
        self.ln_to_aka, self.aka_to_ln = None, None

        self.unique_id = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
        self.save_dir = f"./coverage_cache/{self.unique_id}"

    def load(self, f_name, unpack, params=None):
        self.Coverage = KMNC.load(self.save_dir, f_name, unpack=unpack,
                                  params=params, verbose=False)

    def save(self, prefix="imagenet"):
        os.makedirs(self.save_dir, exist_ok=True)
        self.Coverage.save(self.save_dir, prefix=prefix,
                           aka_ln=self.ln_to_aka, verbose=False)

    def build_nac(self, net: nn.Module, reload=False, prefix="imagenet"):
        unpack = lambda b, device: (b['data'].to(device), b['label'].to(device))
        f_name = prefix
        for ln, ln_hypers in self.layer_kwargs.items():
            ln_prefix = [f"{k}_{v}" for k, v in ln_hypers.items() if k != 'O']
            f_name += f"_{self.ln_to_aka[ln]}_" + "_".join(ln_prefix)
        f_name += f"_states.pkl"
        if reload and os.path.isfile(os.path.join(self.save_dir, f_name)):
            self.load(f_name, unpack, params=self.layer_kwargs)
        else:
            self.Coverage = KMNC(self.layer_size_dict,
                                 hyper=self.layer_kwargs,
                                 unpack=unpack)
            if self.use_cache:
                self.Coverage.assess_with_cache(self.nac_dataloader)
            else:
                self.Coverage.assess(net, self.nac_dataloader,
                                     self.spatial_func)
            self.save(prefix=prefix)
        self.Coverage.update()
        self.build_nac_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, id_name="imagenet",
              valid_num=None, layer_names=None, aps=None, use_cache=False, **kwargs):
        self.model_name = net.__class__.__name__
        if hasattr(net, "backbone"):
            self.model_name = net.backbone.__class__.__name__
            print(self.model_name)
        if valid_num is not None:
            self.valid_num = valid_num
        if layer_names is not None:
            self.layer_names = layer_names
        if aps is not None:
            self.APS_mode = aps
            self.build_nac_flag = not aps
        self.use_cache = use_cache
        self.layer_kwargs = {ln: self.layer_kwargs[ln] for ln in self.layer_kwargs if ln in self.layer_names}
        self.args_dict = {f"{ln}_{k}": v for ln in self.layer_kwargs for k, v in self.args_dict[ln].items()}

        self.aka_to_ln, self.layer_names = get_intr_name(self.layer_names, self.model_name, net)
        self.ln_to_aka = {v: k for k, v in self.aka_to_ln.items()}
        self.layer_kwargs = {self.aka_to_ln[aka]: v for aka, v in self.layer_kwargs.items()}
        print(f"Setup NAC Postprocessor (valid_num:{self.valid_num}, layers:{self.layer_names})......")

        self.spatial_func = StatePooling(self.model_name)
        if self.use_cache:
            self.nac_dataloader = id_loader_dict['main_train']
            dummy_shape = (3, 32, 32) if "cifar" in id_name else (3, 224, 224)
        else:
            self.nac_dataset = TrainSubset(id_loader_dict['main_train'].dataset,
                                           valid_num=self.valid_num,
                                           balanced=True)
            self.nac_dataloader = DataLoader(self.nac_dataset,
                                             batch_size=64, shuffle=False,
                                             num_workers=8, pin_memory=True,
                                             drop_last=False)

            dummy_shape = self.nac_dataset.dataset[0]['data'].shape
        print("Input shape:", dummy_shape)
        self.layer_size_dict = make_layer_size_dict(net, self.layer_names,
                                                    input_shape=(3, *dummy_shape),
                                                    spatial_func=self.spatial_func)
        print(self.layer_size_dict)
        if self.build_nac_flag:
            self.build_nac(net)

    def inference(self, net, data_loader, progress=True):
        if self.use_cache:
            confs, flags, preds, labels = self.Coverage.assess_ood_with_cache(data_loader,
                                                                              progress=progress)
        else:
            confs, flags, preds, labels = self.Coverage.assess_ood(net, data_loader,
                                                                   self.spatial_func,
                                                                   progress=progress)
        return preds, confs, labels



    def set_hyperparam(self, hyperparam: list):
        assert (len(hyperparam) / 4) == len(self.layer_kwargs)
        print("##" * 30)
        i = 0
        for ln in self.layer_kwargs:
            O, M, sig_alpha, method = hyperparam[i:i + 4]
            self.layer_kwargs[ln].update({"O": O, "M": M, "sig_alpha": sig_alpha, "method": method})
            print("Set {} paramters to O:{}, M:{}, sig_alpha:{}, method:{}".format(ln, O, M, sig_alpha, method))
            i = i + 4
        self.build_nac_flag = True

    def get_hyperparam(self):
        print_str = ""
        for ln in self.layer_kwargs:
            print_str += "\n{} paramters O:{}, M:{}, " \
                         "sig_alpha:{}, method:{}".format(ln, *list(self.layer_kwargs[ln].values()))
        return print_str
