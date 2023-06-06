from utilities import log
import torch
import torchvision as tv
import time
import argparse

import numpy as np

from utilities.test_utils import get_measures, AverageMeter
import os, collections

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utilities.mahalanobis_lib import get_Mahalanobis_score
from models.backbones import get_backbone, get_transform

from nac.utils import StatePooling, save_statistic, load_statistic, acc
from nac.instr_state import get_intr_name
from nac.coverage import make_layer_size_dict, NAC
from sconf import Config
from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable


def make_id_ood(args, logger, in_transform=None, out_transform=None, only_in=False):
    """Returns train and validation datasets."""
    if "BiT" in args.model:
        ## BiT model is finetuned with converted labels
        # following https://github.com/deeplearning-wisc/large_scale_ood
        logger.info(f"load customized imagenet dataset (with label conversion)")
        from utilities.dataset import get_dataset_for_bit
        in_set = get_dataset_for_bit(args.in_datadir, in_transform)
    else:
        in_set = tv.datasets.ImageFolder(args.in_datadir, in_transform)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if only_in:
        return in_set, in_loader

    out_set = tv.datasets.ImageFolder(args.out_datadir, out_transform)
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")
    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader


def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 1000 == 0:
            logger.info('{} batches processed'.format(b))

        # debug
        # if b > 500:
        #    break

    return np.array(confs)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def iterate_data_energy(data_loader, model, temper):
    confs = []
    total, correct = 0, 0
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    for b, (x, y) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)
            correct_num, correct_flags = acc(logits, y.cuda())
            total += x.shape[0]
            correct += correct_num
            acc1, acc5 = accuracy(logits, y.cuda(), topk=(1, 5))

            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #       .format(top1=top1, top5=top5))
            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    print("ACC: ", correct / total)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return np.array(confs)


def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor, logger):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 1000 == 0:
            logger.info('{} batches processed'.format(b))
        x = x.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)


def iterate_data_gradnorm(args, data_loader, model, temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 1000 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()
        if 'BiT' in args.model:
            layer_grad = model.head.conv.weight.grad.data
        elif 'resnet' in args.model:
            layer_grad = model.fc.weight.grad.data
        elif 'mobilenet' in args.model:
            layer_grad = model.classifier[-1].weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)



def create_coverage(args, hparams, data_loader, model, layer_size_dict=None,
                    save_name="ImageNet", pooling_func=None):
    unpack = lambda b, device: (b[0].to(device), b[1].to(device))
    prefix = save_name + "_"
    if args.use_thresh:
        prefix += f"thresh_{args.thresh}"
    save_dir = os.path.join(args.logdir, "coverage_cache")
    f_name = prefix + f"_sig_alpha_{hparams.step_kwargs['sig_alpha']}" + \
             f"_states_M{hparams.build_kwargs['M']}" + \
             f"_{hparams.step_kwargs['method']}.pkl"
    if args.reload and os.path.isfile(os.path.join(save_dir, f_name)):
        NACUE = NAC.load(save_dir, f_name, unpack=unpack)
        # NACUE = NAC.load(save_dir, f_name, unpack=unpack, O_star=args.O_star)
    else:
        NACUE = NAC(layer_size_dict,
                    hyper=hparams['build_kwargs'],
                    unpack=unpack)
        _, acc = NACUE.assess(model, data_loader, pooling_func,
                              **hparams['step_kwargs'])
        print("ACC: ", acc)
        os.makedirs(save_dir, exist_ok=True)
        NACUE.save(save_dir, prefix=prefix)
        args.reload = True
        # raise ValueError("We do not create NAC function at this moment")
    NACUE.update(method="avg")
    return NACUE


def iterate_data_coverage(NACUE, data_loader, model, hparams, pooling_func=None):
    # OOD Test
    confs = NACUE.assess_uncertainty(model, data_loader, pooling_func, method=hparams.test_method)
    print("Confidence: ", confs.mean(), confs.shape)
    return confs


def run_eval(dname, model, in_loader, out_loader, logger, args, num_classes,
             hparams=None):
    # switch to evaluate mode
    model.eval()
    logger.info("Running test...")
    logger.flush()
    args.logdir = Path(args.logdir)
    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm(args, in_loader, model, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_gradnorm(args, out_loader, model, args.temperature_gradnorm, num_classes)

    elif args.score == 'Coverage':
        pooling_func = StatePooling(args.model)
        instr_layers = get_intr_name(args)
        layer_size_dict = make_layer_size_dict(model, instr_layers, input_shape=args.input_shape,
                                               pooling_func=pooling_func)
        print(layer_size_dict)
        name = "ImageNet-InD"
        NACUE = create_coverage(args, hparams, in_loader, model, layer_size_dict,
                                save_name=name, pooling_func=pooling_func)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_coverage(NACUE, out_loader, model, hparams, pooling_func=pooling_func)
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_coverage(NACUE, in_loader, model, hparams, pooling_func=pooling_func)
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    if args.save:
        np.save(args.logdir / args.name / "in_scores.npy", in_scores)
        np.save(args.logdir / args.name / f"{dname}_out_scores.npy", out_scores)
    # print(in_scores[:5], out_scores[:5])
    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    return {"AUCROC": auroc,
            "AUCROC (In)": aupr_in,
            "AUPR (Out)": aupr_out,
            "FPR95": fpr95, }


def nac_params_registry(args):
    keys = [open("coverage.yaml", encoding="utf8")]
    hparams = Config(*keys)[args.model]
    if args.sig_alpha is not None:
        hparams.step_kwargs['sig_alpha'] = args.sig_alpha
    if args.O_star is not None:
        hparams.build_kwargs['r'] = args.O_star
    if args.M is not None:
        hparams.build_kwargs['M'] = args.M
    if args.state_method is not None:
        hparams.step_kwargs['method'] = args.state_method
    return Config(hparams)


def main(args):
    logger = log.setup_logger(args)
    if args.score == 'GradNorm':
        args.batch = 1
    input_shape, in_transform, out_transform = get_transform(args.model)
    args.input_shape = input_shape

    hparams = nac_params_registry(args)
    logger.info("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.info("\t" + line)

    # we use ImageNet pretrained model
    model, _ = get_backbone(name=args.model, num_classes=1000, pretrained=True,
                            use_thresh=args.use_thresh, thresh=args.thresh)
    if args.score != 'GradNorm':
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    results = collections.defaultdict(list)
    start_time = time.time()

    logger.info(f"Set Seed: {args.seed}")
    for ood_name in args.out_datasets:
        args.out_datadir = os.path.join(args.out_dataroot, ood_name)
        in_set, out_set, in_loader, out_loader = make_id_ood(args, logger, in_transform, out_transform)
        records = run_eval(ood_name, model, in_loader, out_loader, logger, args,
                           num_classes=len(in_set.classes), hparams=hparams)
        for k, v in records.items():
            results[k].append(v)

    ### add Avg. to results
    end_time = time.time()
    logger.info("Total running time: {}".format(end_time - start_time))

    name = args.score + f'_{args.model}_BATCH({args.batch})'
    if args.score == 'Coverage':
        for k, v in hparams.build_kwargs.items():
            name += f' ({k}:{v})'
        for k, v in hparams.step_kwargs.items():
            name += f' ({k}:{v})'
    if args.use_thresh:
        name += f'_use_thresh: {args.thresh}'

    logger.info('==============Results for {}==============='.format(name))
    table = PrettyTable(["Method"] + args.out_datasets + ["Avg."])
    for key, row in results.items():
        row.append(np.mean(row))
        row = [f"{score:.2%}" for score in row]
        table.add_row([key] + row)

    save_statistic(args.logdir / args.name, results,
                   name=f"s{args.seed}_M{hparams.build_kwargs['M']}" +
                        f"_O_star{hparams.build_kwargs['O_star']}" +
                        f"_{hparams.test_method}" +
                        f"_sig{hparams.step_kwargs['sig_alpha']}.pkl")
    logger.nofmt(table)
    logger.flush()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--logdir", help="Where to log test info (small).")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size.")
    parser.add_argument("--name", help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument("--model", default="BiT-S-R101x1", help="Which variant to use")

    parser.add_argument("--in_datadir", default='/dataset/cv/imagenet/val',
                        help="Path to the in-distribution data folder.")
    parser.add_argument("--out_dataroot", default='/dataset/cv/ood_data/',
                        help="Path to the out-of-distribution data folder.")
    parser.add_argument('--out_datasets', default=['iNaturalist', 'SUN', 'Places', 'Textures'],
                        nargs="*", type=str, help="ood dataset name")
    parser.add_argument('--score', default='Coverage',
                        choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm', 'Coverage'])
    parser.add_argument('--no-save', action="store_true", help="do not save computed scores")
    parser.add_argument('--seed', default=0, type=int, help="random seed")

    # arguments for Coverage
    parser.add_argument('--layer-name', default='avgpool', help="specific layer where neuron resides")
    parser.add_argument('--sig_alpha', default=None, type=float, help='sigmoid steepness')
    parser.add_argument('--O_star', default=None, type=float, help='minimum activated times for covering an interval')
    parser.add_argument('--M', default=None, type=int, help='no. of intervals for NAC estimation')
    parser.add_argument('--state_method', default=None, type=str, help='coverage cal method')

    # arguments for ReAct
    parser.add_argument('--use-thresh', action="store_true", default=False,
                        help="compute stats with thresh (ReAct-NeurIP21)")
    parser.add_argument('--thresh', default=1.0, type=float, help="thresh for rectification")

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')
    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=int,
                        help='temperature scaling for GradNorm')
    args = parser.parse_args()
    args.seed = set_seed(args.seed)

    args.save = not args.no_save
    args.reload = False  # We recompute coverage stats for each program
    main(args)
