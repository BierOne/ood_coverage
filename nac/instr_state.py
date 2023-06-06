"""
Create Time: 24/3/2023
Author: BierOne (lyibing112@gmail.com)
"""
import os, sys
import torch

def get_intr_name(args):
    # if hasattr(model, "module"):
    #     iter_modules = model.module.named_children()
    instr_layers = []
    if 'BiT' in args.model:
        if args.layer_name == "before_head":
            instr_layers = ["module.before_head"]
        elif args.layer_name == "block4":
            instr_layers = ["module.body.block4"]
        elif args.layer_name == "block3":
            instr_layers = ["module.body.block3"]
        elif args.layer_name == "block2":
            instr_layers = ["module.body.block2"]
        elif args.layer_name == "block1":
            instr_layers = ["module.body.block1"]
        else:
            raise ValueError(args.layer_name)
    elif 'resnet' in args.model:
        if args.layer_name == "layer4":
            instr_layers = ["module.layer4"]
        elif args.layer_name == "layer1":
            instr_layers = ["module.layer1"]
        elif args.layer_name == "layer2":
            instr_layers = ["module.layer2"]
        elif args.layer_name == "layer3":
            instr_layers = ["module.layer3"]
        elif args.layer_name == "avgpool":
            instr_layers = ["module.avgpool"]
        elif args.layer_name == "maxpool":
            instr_layers = ["module.maxpool"]
    elif 'mobilenet' in args.model:
        instr_layers = ["module.avgpool"]
    else:
        raise ValueError(args.model)
    return instr_layers


def kl_grad(b_state, outputs, temperature=1.0, retain_graph=False):
    """
    This implementation follows https://github.com/deeplearning-wisc/gradnorm_ood
    """
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    num_classes = outputs.shape[-1]
    targets = torch.ones(outputs.shape).cuda() / num_classes
    loss = (torch.sum(-targets * logsoftmax(outputs), dim=-1))
    # print("kl loss: ", loss[:3])
    layer_grad = torch.autograd.grad(loss.sum(), b_state, create_graph=False,
                                     retain_graph=retain_graph)[0]
    return layer_grad
