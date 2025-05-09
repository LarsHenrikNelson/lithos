from typing import Literal

from scipy.stats import linregress
from scipy.optimize import curve_fit
import numpy as np
from numpy.polynomial import Polynomial
from ..plotting.types import FitFunc


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


def fit_polynomial(x, y, degree=2):
    output = Polynomial.fit(x, y, degree)
    fit_y = output(x)
    return output, fit_y


def fit_linear_regression(x, y):
    output = linregress(x, y)
    fit_y = line(x, output.slope, output.intercept)
    return output, fit_y


def fit_sine(x, y):
    p0 = guess_sine(x, y)
    output = curve_fit(sine, x, y, p0=p0)
    fit_y = sine(x, *output[0])
    return output, fit_y


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
