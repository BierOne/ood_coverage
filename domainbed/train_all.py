import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
# from domainbed.trainer import train
from domainbed.trainer_nac import train


def main():
    parser = argparse.ArgumentParser(description="Domain generalization", allow_abbrev=False)
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    parser.add_argument("--no-stopping", action="store_true", help="not stop early when using swad")
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--no-save-best', action='store_true')
    parser.add_argument('--compute-coverage', action='store_true')
    parser.add_argument('--out_dir', type=str, default="/data/lyb/dg/dense_benchmark")
    parser.add_argument('--workers', type=int, default=None, help="dataset workers")
    parser.add_argument('--single_train', action="store_true", help="training with single domain")
    parser.add_argument('--save-suffix', type=str, default="")
    parser.add_argument('--sweep', action="store_true", help="sweep params")

    # parser.add_argument('--model', type=str, default="resnet50",
    #                     choices=["resnet18", "resnet50", "swag_regnety_16gf",
    #                              "clip_resnet", "clip_vit-b16"])
    # parser.add_argument('--pretrained', default=True)

    args, left_argv = parser.parse_known_args()
    args.save_best = not args.no_save_best
    args.deterministic = True
    args.save_suffix = "" if "mlp" not in args.save_suffix else args.save_suffix  # default use ""

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    # load default param
    keys = [open(key, encoding="utf8") for key in args.configs]
    alg_type = args.algorithm.split('_')[0]
    alg_params = Config(*keys, default={})[f"{alg_type}_{args.dataset}_params"]
    hparams.update(alg_params)

    keys = [open("config.yaml", encoding="utf8")]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.data_dir = Path(args.data_dir)

    suffix = ""
    if hparams.swad:
        suffix = '-swad'
    if not hparams.pretrained:
        suffix += "-init"

    args.out_dir = Path(args.out_dir) / (hparams.model+suffix) / args.algorithm / args.dataset
    if args.sweep:
        args.out_dir = args.out_dir / f"lr_{hparams['lr']}_wd_{hparams['weight_decay']}_drop_{hparams['resnet_dropout']}"

    args.out_dir = args.out_dir / f'trial_{args.trial_seed}'
    args.out_dir.mkdir(exist_ok=True, parents=True)

    writer = get_writer(args.out_dir / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dummy datasets for logging information.
    # Real dataset will be re-assigned in train function.
    # test_envs only decide transforms; simply set to zero.
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)

    # print dataset information
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")
    logger.nofmt("")

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if args.single_train:
        # training with single domain, and test on others
        all_envs = list(range(len(dataset)))
        args.test_envs = []
        for tr_env in all_envs:
            te_env = [e for e in all_envs if e != tr_env]
            args.test_envs.append(te_env)
    else:
        if not args.test_envs:
            args.test_envs = [[te] for te in range(len(dataset))]
        else:
            args.test_envs = [[te] for te in args.test_envs]

    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    all_records = []
    results = collections.defaultdict(list)

    for test_env in args.test_envs:
        args.save_dir = args.out_dir / '_'.join(map(str, test_env))
        args.save_dir.mkdir(exist_ok=True, parents=True)

        res, records = train(
            test_env,
            args=args,
            hparams=hparams,
            n_steps=n_steps,
            checkpoint_freq=checkpoint_freq,
            logger=logger,
            writer=writer,
        )
        all_records.append(records)
        for k, v in res.items():
            results[k].append(v)

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)


    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])

    for key, row in results.items():
        row.append(np.mean(row))
        row = [f"{acc:.3%}" for acc in row]
        table.add_row([key] + row)
    logger.nofmt(table)


if __name__ == "__main__":
    main()
