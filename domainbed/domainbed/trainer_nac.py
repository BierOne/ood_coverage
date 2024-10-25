import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.swad import swad as swad_module
from domainbed.swad import swa_utils

from benchmark_notes import utils
from benchmark_notes.utils import FeatSpatialize, save_statistic
from benchmark_notes.instr_state import get_intr_name
from benchmark_notes.coverage import make_layer_size_dict
from benchmark_notes.metrics import rank_correlation

def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def save_checkpoint(args, algorithm, hparams, filename, tgt_acc=0.0, out_train_acc_dict=None,
                    out_test_acc_dict=None, coverage=None):
    save_dict = {
        "args": vars(args),
        "model_hparams": dict(hparams),
        "model_dict": algorithm.state_dict(),
        "test_acc": tgt_acc,
        "out_train_acc_dict": out_train_acc_dict,
        "out_test_acc_dict": out_test_acc_dict,
        "coverage": coverage,
    }
    torch.save(save_dict, args.save_dir / filename)


def subdict_by_filtering(result_dict, names):
    sub_dict = {k: result_dict[k] for k in result_dict if k in names}
    return sub_dict

def compute_rc(records):
    ### Compute Rank Correlation
    coverage_scores = records.select(lambda rs: rs['coverage'])
    test_scores = records.select(lambda rs: rs['test_in'])
    val_scores = records.select(lambda rs: rs['train_out'])
    oracle_scores = records.select(lambda rs: rs['test_out'])
    return {
        "coverage_rc": rank_correlation(coverage_scores, test_scores),
        "val_rc": rank_correlation(val_scores, test_scores),
        "oracle_rc": rank_correlation(oracle_scores, test_scores),
        "test": 1.0
    }

def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )

    algorithm.cuda()

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    val_best, test_best = 0, 0
    m_coverage, training_statistics = {}, {}
    out_train_names = ['env{}_out_acc'.format(i) for i in range(len(out_splits)) if i not in test_envs]
    out_test_names = ['env{}_out_acc'.format(i) for i in range(len(out_splits)) if i in test_envs]
    save_func = lambda name, alg, tgt_acc, results, cov: save_checkpoint(args, alg, hparams, name, tgt_acc,
                                                                         out_test_acc_dict=subdict_by_filtering(
                                                                             results, out_test_names),
                                                                         out_train_acc_dict=subdict_by_filtering(
                                                                             results, out_train_names),
                                                                         coverage=cov
                                                                         )
    spatial_func = FeatSpatialize(hparams.model)
    instr_layers = get_intr_name(hparams['model'], args.algorithm, args.dataset)
    layer_size_dict = make_layer_size_dict(algorithm, instr_layers, spatial_func=spatial_func)
    print(layer_size_dict)

    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
        get_coverage=hparams.coverage,
        layer_size_dict=layer_size_dict,
        coverage_hyper=hparams.coverage_kwargs,
    )

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, "LossValley")
        swad = swad_cls(evaluator, **hparams.swad_kwargs)
    swad_update_flag = True

    last_results_keys = None
    records = []
    epochs_path = args.save_dir / "results.jsonl"


    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device
        batches = {
            key: [tensor.cuda() for tensor in tensorlist] for key, tensorlist in batches.items()
        }

        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if swad and swad_update_flag:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            if hparams.coverage:
                m_coverage, states, tr_accs = evaluator.evaluate_coverage(algorithm, step=step,
                                                                          use_train=not (args.dataset=="DomainNet"),
                                                                          save_dir=args.save_dir/(args.save_suffix+"coverage_cache"),
                                                                          spatial_func=spatial_func,
                                                                          **hparams.coverage_kwargs['step_kwargs'])
                # utils.save_statistic(args.save_dir, states, name=f"state_statistics_step{step}")
                accuracies.update(tr_accs)
                results['coverage'] = m_coverage[hparams.coverage_kwargs['report_name']][-1].item()

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            if args.save_best:
                if summaries['test_out'] > test_best:
                    test_best = summaries['test_out']
                    print("save best with test out acc (oracle)", test_best)
                    save_func('model_best_oracle.pkl', algorithm, summaries['test_in'], results, m_coverage)

                if summaries['train_out'] > val_best:
                    val_best = summaries['train_out']
                    print("save best with val acc", val_best)
                    save_func('model_best_val.pkl', algorithm, summaries['test_in'], results, m_coverage)

            m_name = f'model_step{step}'
            if args.save_model_every_checkpoint:
                save_func(m_name + '.pkl', algorithm, summaries['test_in'], results, m_coverage)
            training_statistics[m_name] = ({},subdict_by_filtering(results, out_train_names),
                                           subdict_by_filtering(results, out_test_names), summaries['test_in'])

            # swad
            if swad and swad_update_flag:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["train_out"], results["tr_outloss"], prt_results_fn
                )
 
                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    swad_update_flag = False
                    logger.info("SWAD valley is dead -> early stop !")
                    if args.no_stopping:
                        logger.info("We will continue the training, but will not update swad_algorithm anymore")
                        continue
                    else:
                        break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset
            logger.info("---")
            print(compute_rc(Q(records)))

        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")

    # find best
    logger.info("---")
    save_statistic(args.save_dir, training_statistics, name="training_statistics")

    records = Q(records)
    oracle_best = records.argmax("test_out")["test_in"]
    tr_val_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    in_key = "train_out"
    tr_val_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    # NOTE for clearity, report only training-domain validation results.
    ret = {
        # "last (inD)": last_indomain,
        # "training-domain validation (inD)": tr_val_best_indomain,
        "oracle": oracle_best,
        "last": last,
        "training-domain validation": tr_val_best,
    }

    if hparams.coverage:
        coverage_best = records.argmax("coverage")["test_in"]
        ret.update({"coverage": coverage_best})
        ret.update(compute_rc(records))

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records
