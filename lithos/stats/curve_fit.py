from typing import Literal

from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
from numpy.polynomial import Polynomial
from ..plotting.types import FitFunc, CIFunc


def confidence_intervals(x, y, fit_x, residuals):
    n = len(y)
    dof = n - 2
    t = stats.t.ppf(0.975, dof)
    s_err = np.sqrt(np.sum(residuals**2) / dof)
    ci = (
        t
        * s_err
        * np.sqrt(1 / n + (fit_x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )
    return ci


def boostrap_confidence_intervals(x, y, fit_x):
    rng = np.random.default_rng(seed=42)
    boot_dist = []
    n_boot = 100
    args = [x, y]
    n = len(y)
    for i in range(int(n_boot)):
        resampler = rng.integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [np.take(a, resampler, axis=0) for a in args]
        boot_dist.append(stats.linregress(*sample))
    yfits = [output.slope * fit_x + output.intercept for output in boot_dist]
    ci = np.percentile(np.array(yfits), q=[97.5, 2.5], axis=0)
    yhat = np.mean(yfits, axis=0)
    ci = ci - yhat
    ci[1] *= -1
    return ci


def prediction_intervals(x, y, fit_x, residuals):
    n = len(y)
    dof = n - 2
    t = stats.t.ppf(0.975, dof)
    s_err = np.sqrt(np.sum(residuals**2) / dof)
    pi = (
        t
        * s_err
        * np.sqrt(1 + 1 / n + (fit_x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )
    return pi


def get_ci_func(ci_func: CIFunc = "ci", **kwargs):
    if ci_func == "ci":
        return confidence_intervals(
            x=kwargs["x"],
            y=kwargs["y"],
            fit_x=kwargs["fit_x"],
            residuals=kwargs["residuals"],
        )
    elif ci_func == "pi":
        return prediction_intervals(
            x=kwargs["x"],
            y=kwargs["y"],
            fit_x=kwargs["fit_x"],
            residuals=kwargs["residuals"],
        )
    elif ci_func == "bootstrap_ci":
        return boostrap_confidence_intervals(
            x=kwargs["x"], y=kwargs["y"], fit_x=kwargs["fit_x"]
        )
    else:
        return None


def sine(x, amplitude=1.0, omega=1.0, phase=0.0, offset=0.0):
    return amplitude * np.sin(omega * x + phase) + offset


def guess_sine(x, y):
    x = np.array(x)
    y = np.array(y)
    ff = np.fft.fftfreq(len(x), (x[1] - x[0]))  # assume uniform spacing
    Fy = abs(np.fft.fft(y))
    guess_freq = abs(
        ff[np.argmax(Fy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.0**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])
    return guess


def line(x, slope=1.0, intercept=0.0):
    return slope * x + intercept


def fit_polynomial(x, y, fit_x=None, degree=2):
    output = Polynomial.fit(x, y, degree)
    if fit_x is None:
        fit_x = np.sort(x)
    fit_y = output(fit_x)
    residuals = y - output(x)
    ci = get_ci_func(x, y, fit_x, residuals, ci_func=None)
    return output, fit_y, fit_x, ci


def fit_linear_regression(x, y, fit_x=None, ci_func: CIFunc = "ci"):
    if len(y) == 1:
        return None, y, x, [0]
    output = stats.linregress(x, y)
    if fit_x is None:
        fit_x = np.sort(x)
    fit_y = line(fit_x, output.slope, output.intercept)
    residuals = y - line(x, output.slope, output.intercept)
    ci = get_ci_func(x=x, y=y, fit_x=fit_x, residuals=residuals, ci_func=ci_func)
    return output, fit_y, fit_x, ci


def fit_sine(x, y, fit_x=None, ci_func: CIFunc = "ci"):
    p0 = guess_sine(x, y)
    output = curve_fit(sine, x, y, p0=p0)
    if fit_x is None:
        fit_x = np.sort(x)
    fit_y = sine(fit_x, *output[0])
    residuals = y - line(x, output.slope, output.intercept)
    ci = get_ci_func(x=x, y=y, fit_x=fit_x, residuals=residuals, ci_func=ci_func)
    return output, fit_y, fit_x, ci


FIT_DICT = {
    "linear": fit_linear_regression,
    "sine": fit_sine,
    "polynimial": fit_polynomial,
}


def fit(fit_func: FitFunc, **kwargs):
    if fit_func in FIT_DICT:
        output = FIT_DICT[fit_func](**kwargs)
    else:
        output = fit_func(**kwargs)
    return output
