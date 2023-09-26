"""
Create Time: 24/3/2023
Author: BierOne (lyibing112@gmail.com)
"""
import os, sys
from collections import OrderedDict
import torch


def get_intr_name(layer_names, model_name, network=None):
    aka_name_dict = OrderedDict()
    for aka_layer in layer_names:
        layer = aka_layer.split('.')[-1]
        if 'resnet' in model_name.lower():
            assert layer in ["layer1", "layer2", "layer3", "layer4", "conv1",
                             "avgpool", "fc", "maxpool"]
            if layer in ["avgpool", "fc", "maxpool", "conv1"]:
                aka_name_dict[aka_layer] = layer
            elif layer in ["layer1", "layer4"]:
                aka_name_dict[aka_layer] = layer
            elif layer in ["layer2"]:
                aka_name_dict[aka_layer] = layer
            elif layer in ["layer3"]:
                aka_name_dict[aka_layer] = layer

        elif 'vit' in model_name.lower():
            layer_list = [f"layer_{i}" for i in range(12)]
            if 'conv_proj' == layer:
                aka_name_dict[aka_layer] = layer
            elif 'layer_norm' == layer:
                aka_name_dict[aka_layer] = "encoder.ln"
            elif layer in layer_list:
                aka_name_dict[aka_layer] = f"encoder.layers.encoder_{layer}.ln_2"
            else:
                raise ValueError(layer)

        elif 'swin' in model_name.lower():
            shallow_layer_list = [f"layer_{i}" for i in [1, 3, 7]]
            deep_layer_list = [f"layer_{i}" for i in [5]]

            patch_list = [f"patch_{i}" for i in [2, 4, 6]]
            if 'conv_proj' == layer:
                aka_name_dict[aka_layer] = "features.0"
            elif 'avgpool' == layer:
                aka_name_dict[aka_layer] = "avgpool"
            elif 'layer_norm' == layer:
                aka_name_dict[aka_layer] = "norm"
            elif layer in shallow_layer_list:
                aka_name_dict[aka_layer] = f"features.{layer.split('_')[-1]}.1.mlp.0"
            elif layer in deep_layer_list:
                aka_name_dict[aka_layer] = f"features.{layer.split('_')[-1]}.5.mlp.0"
            elif layer in patch_list:
                aka_name_dict[aka_layer] = f"features.{layer.split('_')[-1]}"
            else:
                raise ValueError(layer)
        else:
            raise ValueError(model_name.lower())

    if hasattr(network, "backbone"):
        aka_name_dict = {aka: "backbone." + ln for aka, ln in aka_name_dict.items()}
    # if hasattr(network, "module"):
    #     aka_name_dict = {aka: "module." + ln for aka, ln in aka_name_dict.items()}
    return aka_name_dict, list(aka_name_dict.values())


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
