from typing import Literal

import numpy as np
from numpy.polynomial import Polynomial
from scipy import stats
from scipy.optimize import curve_fit

from ..types.basic_types import CIFunc, FitFunc


def _validate_fit(x, y, n_params: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate fit inputs: equal lengths and more points than fit parameters.

    Args:
        x: Independent values.
        y: Dependent values.
        n_params (int): Number of fit parameters.
        name (str): Fit name used in error messages.

    Returns:
        tuple[np.ndarray, np.ndarray]: The flattened float x/y arrays.

    Raises:
        ValueError: If x and y differ in length, or the number of points is
            not greater than the number of fit parameters.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    if xv.size != yv.size:
        raise ValueError(f"{name} fit requires x and y of the same length, got {xv.size} and {yv.size}.")
    if xv.size <= n_params:
        raise ValueError(f"{name} fit requires more points than fit parameters ({n_params}), got {xv.size}.")
    return xv, yv


def _fit_n_params(x, n_params: int) -> int:
    """Effective parameter count: all-identical x collapses the fit to the mean."""
    x = np.asarray(x, dtype=float)
    return 1 if np.ptp(x) == 0 else n_params


def _fit_polynomial_model(x, y, degree: int) -> Polynomial:
    """Fit a polynomial, collapsing all-identical x to slope 0 / intercept mean.

    With all-identical x the least-squares fit is well-defined: slope 0 with
    the mean of y as intercept — a constant line (a degree-0 fit at any
    requested degree).

    Args:
        x: Independent values.
        y: Dependent values.
        degree (int): Polynomial degree.

    Returns:
        Polynomial: The fitted polynomial (constant-x coefficients in the
        standard basis).
    """
    x = np.asarray(x, dtype=float)
    if np.ptp(x) == 0:
        coef = np.zeros(degree + 1)
        coef[0] = np.mean(y)
        return Polynomial(coef)
    return Polynomial.fit(x, y, degree)


def confidence_intervals(x, y, fit_x, residuals, n_params=2):
    """Two-sided 95% confidence interval half-widths around a fit.

    Args:
        x: Independent values used for the fit.
        y: Dependent values used for the fit.
        fit_x: Grid the interval is evaluated on.
        residuals: Fit residuals (``y - fitted(x)``).
        n_params (int): Number of fit parameters; the degrees of freedom are
            ``n - n_params``. All-identical ``x`` collapses the fit to a
            single mean parameter, and the band becomes the mean's CI.

    Returns:
        np.ndarray: CI half-widths, one per ``fit_x`` value (a constant band
        for all-identical ``x``).
    """
    x = np.asarray(x, dtype=float)
    n = len(y)
    dof = n - _fit_n_params(x, n_params)
    t = stats.t.ppf(0.975, dof)
    s_err = np.sqrt(np.sum(residuals**2) / dof)
    sxx = np.sum((x - np.mean(x)) ** 2)
    fit_x = np.asarray(fit_x, dtype=float)
    if sxx == 0:
        # constant x: the fit is the mean and the interval is the mean's CI
        return np.full_like(fit_x, t * s_err / np.sqrt(n))
    ci = t * s_err * np.sqrt(1 / n + (fit_x - np.mean(x)) ** 2 / sxx)
    return ci


def bootstrap_confidence_intervals(x, y, fit_x, degree=1):
    """Percentile bootstrap CI of a polynomial fit around the mean bootstrap fit.

    Resample x/y pairs, refit the polynomial model on each resample, and take
    the 95% percentile envelope around the mean bootstrap fit (legacy
    ``ci_func="bootstrap_ci"``). Constant-x resamples collapse to the
    resampled mean.

    Args:
        x: Independent values used for the fit.
        y: Dependent values used for the fit.
        fit_x: Grid the interval is evaluated on.
        degree (int): Polynomial degree (1 = the legacy linear bootstrap).

    Returns:
        np.ndarray: ``(low, high)`` CI half-widths per ``fit_x`` value.
    """
    rng = np.random.default_rng(seed=42)
    n_boot = 100
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fit_x = np.asarray(fit_x, dtype=float)
    n = len(y)
    yfits = []
    for _ in range(int(n_boot)):
        resampler = rng.integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample_x = np.take(x, resampler, axis=0)
        sample_y = np.take(y, resampler, axis=0)
        yfits.append(_fit_polynomial_model(sample_x, sample_y, degree)(fit_x))
    yfits = np.asarray(yfits)
    ci = np.percentile(yfits, q=[97.5, 2.5], axis=0)
    yhat = np.mean(yfits, axis=0)
    ci = ci - yhat
    ci[1] *= -1
    return ci


def prediction_intervals(x, y, fit_x, residuals, n_params=2):
    """Two-sided 95% prediction interval half-widths around a fit.

    Same contract as :func:`confidence_intervals`, with the extra ``+ 1``
    under the root (the scatter of a single new observation); constant x
    gives the mean's prediction interval.

    Args:
        x: Independent values used for the fit.
        y: Dependent values used for the fit.
        fit_x: Grid the interval is evaluated on.
        residuals: Fit residuals (``y - fitted(x)``).
        n_params (int): Number of fit parameters; the degrees of freedom are
            ``n - n_params`` (1 for all-identical ``x``).

    Returns:
        np.ndarray: Prediction interval half-widths, one per ``fit_x`` value
        (a constant band for all-identical ``x``).
    """
    x = np.asarray(x, dtype=float)
    n = len(y)
    dof = n - _fit_n_params(x, n_params)
    t = stats.t.ppf(0.975, dof)
    s_err = np.sqrt(np.sum(residuals**2) / dof)
    sxx = np.sum((x - np.mean(x)) ** 2)
    fit_x = np.asarray(fit_x, dtype=float)
    if sxx == 0:
        # constant x: the fit is the mean and the interval is the mean's prediction interval
        return np.full_like(fit_x, t * s_err * np.sqrt(1 + 1 / n))
    pi = t * s_err * np.sqrt(1 + 1 / n + (fit_x - np.mean(x)) ** 2 / sxx)
    return pi


def get_ci_func(ci_func: CIFunc | None = "ci", **kwargs):
    """Dispatch to the interval function selected by ``ci_func`` (``None`` for no band)."""
    if ci_func == "ci":
        return confidence_intervals(
            x=kwargs["x"],
            y=kwargs["y"],
            fit_x=kwargs["fit_x"],
            residuals=kwargs["residuals"],
            n_params=kwargs.get("n_params", 2),
        )
    elif ci_func == "pi":
        return prediction_intervals(
            x=kwargs["x"],
            y=kwargs["y"],
            fit_x=kwargs["fit_x"],
            residuals=kwargs["residuals"],
            n_params=kwargs.get("n_params", 2),
        )
    elif ci_func == "bootstrap_ci":
        if "degree" not in kwargs:
            raise ValueError("bootstrap_ci is only supported for linear and polynomial fits.")
        return bootstrap_confidence_intervals(
            x=kwargs["x"], y=kwargs["y"], fit_x=kwargs["fit_x"], degree=kwargs["degree"]
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
    guess_freq = abs(ff[np.argmax(Fy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.0**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])
    return guess


def line(x, slope=1.0, intercept=0.0):
    return slope * x + intercept


def fit_polynomial(x, y, fit_x=None, degree=2, ci_func: CIFunc = "ci"):
    """Fit a polynomial of the given degree and evaluate it (with an interval).

    Args:
        x: Independent values.
        y: Dependent values.
        fit_x: Grid the fit is evaluated on; ``None`` uses the sorted x values.
        degree (int): Polynomial degree (2 by default).
        ci_func (CIFunc): Interval around the fit, or ``None`` for no band.

    Returns:
        tuple: ``(Polynomial, fit_y, fit_x, ci)`` — the fitted polynomial, its
        values on the grid, the grid, and the interval half-widths (``None``
        without a ``ci_func``).

    Raises:
        ValueError: If x and y differ in length, the number of points is not
            greater than ``degree + 1``, or ``ci_func`` is unsupported.
    """
    x, y = _validate_fit(x, y, degree + 1, "polynomial")
    output = _fit_polynomial_model(x, y, degree)
    fit_x = np.sort(x) if fit_x is None else np.asarray(fit_x, dtype=float)
    fit_y = output(fit_x)
    residuals = y - output(x)
    ci = get_ci_func(ci_func, x=x, y=y, fit_x=fit_x, residuals=residuals, n_params=degree + 1, degree=degree)
    return output, fit_y, fit_x, ci


def fit_linear_regression(x, y, fit_x=None, ci_func: CIFunc = "ci"):
    """Fit a line as a degree-1 polynomial (slope/intercept from the coefficients).

    All-identical x collapses to slope 0 / intercept ``mean(y)`` with the
    mean's interval (see :func:`confidence_intervals`).

    Args:
        x: Independent values.
        y: Dependent values.
        fit_x: Grid the fit is evaluated on; ``None`` uses the sorted x values.
        ci_func (CIFunc): Interval around the fit, or ``None`` for no band.

    Returns:
        tuple: ``(Polynomial, fit_y, fit_x, ci)`` — same contract as
        :func:`fit_polynomial` with ``degree=1``.

    Raises:
        ValueError: If x and y differ in length, the number of points is not
            greater than 2, or ``ci_func`` is unsupported.
    """
    return fit_polynomial(x=x, y=y, fit_x=fit_x, degree=1, ci_func=ci_func)


def fit_sine(x, y, fit_x=None, ci_func: CIFunc = "ci"):
    """Fit a sine (amplitude, omega, phase, offset) via scipy curve_fit.

    Args:
        x: Independent values.
        y: Dependent values.
        fit_x: Grid the fit is evaluated on; ``None`` uses the sorted x values.
        ci_func (CIFunc): Interval around the fit, or ``None`` for no band
            (``"bootstrap_ci"`` is unsupported: the bootstrap refits a
            polynomial model).

    Returns:
        tuple: ``(popt, fit_y, fit_x, ci)`` — the optimal sine parameters, the
        fitted values on the grid, the grid, and the interval half-widths
        (``None`` without a ``ci_func``).

    Raises:
        ValueError: If x and y differ in length, the number of points is not
            greater than 4, all x values are identical (the FFT-based initial
            guess needs x spacing), or ``ci_func`` is unsupported.
    """
    x, y = _validate_fit(x, y, 4, "sine")
    if np.ptp(x) == 0:
        raise ValueError("sine fit requires variation in x; all x values are identical.")
    p0 = guess_sine(x, y)
    popt, _ = curve_fit(sine, x, y, p0=p0)
    fit_x = np.sort(x) if fit_x is None else np.asarray(fit_x, dtype=float)
    fit_y = sine(fit_x, *popt)
    residuals = y - sine(x, *popt)
    ci = get_ci_func(ci_func, x=x, y=y, fit_x=fit_x, residuals=residuals, n_params=4)
    return popt, fit_y, fit_x, ci


FIT_DICT = {
    "linear": fit_linear_regression,
    "sine": fit_sine,
    "polynomial": fit_polynomial,
}


def fit(fit_func: FitFunc, **kwargs):
    if fit_func in FIT_DICT:
        output = FIT_DICT[fit_func](**kwargs)
    else:
        output = fit_func(**kwargs)
    return output
