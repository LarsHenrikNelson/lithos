from typing import Literal

from scipy import stats
import numpy as np

from ..stats import periodic_mean, periodic_std, periodic_sem


def round_sig(x, sig=2):
    if np.isnan(x):
        return np.nan
    elif x == 0:
        return 0
    elif x != 0 or not np.isnan(x):
        temp = np.floor(np.log10(abs(x)))
        if np.isnan(temp):
            return round(x, 0)
        else:
            return round(x, sig - int(temp) - 1)


def sem(a, axis=None):
    if len(a.shape) == 2:
        shape = a.shape[0]
    else:
        shape = a.size
    denominator = np.sqrt(shape - 1) if shape > 1 else 1
    return np.std(a, axis=axis) / denominator


def ci(a, axis=None):
    if a.ndim == 2:
        length = a.shape[1] - 1
    else:
        length = len(a) - 1
    t_critical = stats.t.ppf(1 - 0.05 / 2, length)
    margin_of_error = t_critical * (np.std(a, ddof=1, axis=axis) / np.sqrt(length))
    return margin_of_error


def ci_bca(a):
    res = stats.bootstrap(a, np.mean)
    print(res.confidence_interval)
    return np.array([[res.confidence_interval.high], [res.confidence_interval.low]])


def mad(a, axis=None):
    return np.median(np.abs(a - np.median(a, axis=axis)))


TRANSFORM = Literal["log10", "log2", "ln", "inverse", "ninverse", "sqrt"]
AGGREGATE = Literal[
    "mean", "periodic_mean", "nanmean", "median", "nanmedian", "gmean", "hmean"
]
ERROR = Literal[
    "sem",
    "ci",
    "periodic_std",
    "periodic_sem",
    "std",
    "nanstd",
    "var",
    "nanvar",
    "mad",
    "gstd",
]

BACK_TRANSFORM_DICT = {
    "log10": lambda x: 10.0**x,
    "log2": lambda x: 2.0**x,
    "ninverse": lambda x: -1.0 / x,
    "inverse": lambda x: 1.0 / x,
    "ln": lambda x: np.e**x,
    "sqrt": lambda x: x**2,
}

FUNC_DICT = {
    "sem": sem,
    "ci": ci,
    "ci_bca": ci_bca,
    "mean": np.mean,
    "periodic_mean": periodic_mean,
    "periodic_std": periodic_std,
    "periodic_sem": periodic_sem,
    "nanmean": np.nanmean,
    "nanmedian": np.nanmedian,
    "median": np.median,
    "std": np.std,
    "nanstd": np.nanstd,
    "log10": np.log10,
    "log2": np.log2,
    "ln": np.log,
    "var": np.var,
    "nanvar": np.nanvar,
    "inverse": lambda a, axis=None: 1 / (a + 1e-10),
    "ninverse": lambda a, axis=None: -1 / (a + 1e-10),
    "sqrt": np.sqrt,
    "mad": mad,
    "wrap_pi": lambda a: np.where(a < 0, a + 2 * np.pi, a),
    "zscore": lambda a: (a - np.mean(a) / np.std(a)),
    "gmean": stats.gmean,
    "hmean": stats.hmean,
    "gstd": stats.gstd,
}


def get_transform(input):
    if input in FUNC_DICT:
        return FUNC_DICT[input]
    elif callable(input):
        return input
    else:
        return lambda a, axis=None: a


def get_backtransform(input):
    if input in BACK_TRANSFORM_DICT:
        return BACK_TRANSFORM_DICT[input]
    elif callable(input):
        return input
    else:
        return lambda a, axis=None: a
