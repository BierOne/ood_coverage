"""
Create Time: 11/4/2023
Author: BierOne (lyibing112@gmail.com)
"""
import torch
import numpy as np

def rank_correlation(map1, map2, map_size=None):
    """
    Function that measures Spearman’s correlation coefficient between two maps

    To evaluate the metric:
            from scipy import stats
            x, y = np.random.randn(10), np.random.randn(10)
            print(stats.spearmanr(x, y))
            print(compute_rank_correlation(x, y))
    """
    def _rank_correlation_(map1_rank, map2_rank, map_size):
        n = np.array(map_size)
        upper = 6 * np.sum(np.power(map2_rank - map1_rank, 2), axis=-1)
        down = n * (np.power(n, 2) - 1.0)
        return 1.0 - (upper / down)

    map1 = np.array(map1)
    map2 = np.array(map2)
    if map_size is None:
        map_size = map1.shape[-1]
    # get the rank for each element, we use sort function two times
    map1 = map1.argsort(axis=-1)
    map1_rank = map1.argsort(axis=-1)

    map2 = map2.argsort(axis=-1)
    map2_rank = map2.argsort(axis=-1)
    rc = _rank_correlation_(map1_rank, map2_rank, map_size)
    return rc


def pearson_correlation(map1, map2):
    """
    Function that measures Pearson’s correlation coefficient between two maps
    """
    map1 = np.array(map1)
    map2 = np.array(map2)

    # Compute Pearson correlation
    correlation_matrix = np.corrcoef(map1, map2)
    # The correlation coefficient is the off-diagonal element
    pearson_corr = correlation_matrix[0, 1]
    return round(pearson_corr, 4)

def compute_pearson_correlation(method_coverage, training_statistics, key='micro_avg'):
    # assert list(method_coverage.keys()) == list(training_statistics.keys())
    cov, val, oracle, test = [], [], [], []
    for k in training_statistics:
        cov.append(method_coverage[k][key][-1])
        val.append(np.mean(list(training_statistics[k][1].values())))
        oracle.append(np.mean(list(training_statistics[k][2].values())))
        test.append(training_statistics[k][-1])
    rc_dict = {
        "coverage": pearson_correlation(cov, test),
        "coverage_val": pearson_correlation(cov, val),
        "val": pearson_correlation(val, test),
        "test_out": pearson_correlation(oracle, test),
        "test": 1.0
    }
    return rc_dict


def compute_rank_correlation(method_coverage, training_statistics, key='micro_avg'):
    # assert list(method_coverage.keys()) == list(training_statistics.keys())
    cov, val, oracle, test = [], [], [], []
    for k in training_statistics:
        cov.append(method_coverage[k][key][-1])
        val.append(np.mean(list(training_statistics[k][1].values())))
        oracle.append(np.mean(list(training_statistics[k][2].values())))
        test.append(training_statistics[k][-1])
    rc_dict = {
        "coverage": rank_correlation(cov, test),
        "coverage_val": rank_correlation(cov, val),
        "val": rank_correlation(val, test),
        "test_out": rank_correlation(oracle, test),
        "test": 1.0
    }
    return rc_dict
