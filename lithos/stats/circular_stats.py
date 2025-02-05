from typing import Union

import numpy as np
from numba import njit
from numpy.random import default_rng

"""
See: https://github.com/aarchiba/kuiper
See: https://docs.astropy.org/en/stable/index.html
"""

__all__ = [
    "h_test",
    "periodic_mean",
    "periodic_sem",
    "periodic_std",
    "ppc_numba",
    "ppc_sampled",
    "rayleightest",
]


def mean_vector_length(spike_phases):
    a_cos = np.cos(spike_phases)
    a_sin = np.sin(spike_phases)
    uv_x = sum(a_cos) / len(a_cos)
    uv_y = sum(a_sin) / len(a_sin)
    uv_radius = np.sqrt((uv_x * uv_x) + (uv_y * uv_y))
    p_value = np.exp(-1 * len(spike_phases) * (uv_radius**2))
    return uv_radius, p_value


@njit(cache=True)
def ppc_numba(spike_phases):
    outer_sums = np.zeros(spike_phases.size - 1)
    array1 = np.zeros(2)
    array2 = np.zeros(2)
    for index1 in range(0, spike_phases.size - 1):
        temp_sum = np.zeros(spike_phases.size - index1 + 1)
        array1[0] = np.cos(spike_phases[index1])
        array1[1] = np.sin(spike_phases[index1])
        for index2 in range(index1 + 1, spike_phases.size):
            array2[0] = np.cos(spike_phases[index2])
            array2[1] = np.sin(spike_phases[index2])
            dp = np.dot(array1, array2)
            temp_sum[index2 - index1] = dp
        outer_sums[index1] = temp_sum.sum()
    dp_sum = np.sum(outer_sums)
    ppc_output = dp_sum / (len(spike_phases) * (len(spike_phases) - 1) // 2)
    return ppc_output


def ppc_sampled(spike_phases, size, iterations, seed=42):
    """This

    Args:
        spike_phases (_type_): _description_
        size (_type_): _description_
        iterations (_type_): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    rng = default_rng(seed)
    output_array = np.zeros(iterations)
    for i in range(iterations):
        spk_sampled = np.ascontiguousarray(
            rng.choice(spike_phases, size=size, replace=False)
        )
        output_array[i] = ppc_numba(spk_sampled)
    return output_array.mean()


def periodic_mean(angles: np.ndarray, axis=None) -> float:
    angles = np.asarray(angles).flatten()
    sines = np.sin(angles)
    cosines = np.cos(angles)
    mean = np.arctan2(np.mean(sines), np.mean(cosines))
    return mean


def periodic_std(angles: np.ndarray, axis=None) -> float:
    angles = np.asarray(angles).flatten()
    sines = np.sin(angles)
    cosines = np.cos(angles)
    R = np.sqrt(np.sum(sines) ** 2 + np.sum(cosines) ** 2) / len(angles)
    std = np.sqrt(-2 * np.log(R))
    return std


def periodic_sem(angles: np.ndarray, axis=None) -> float:
    std = periodic_std(angles)
    return std / np.sqrt(std.size - 1)


@njit(cache=True)
def rayleightest(data: np.ndarray) -> float:
    n = data.size
    S = np.sum(np.sin(data)) / n
    C = np.sum(np.cos(data)) / n
    Rbar = np.hypot(S, C)
    z = n * Rbar * Rbar

    # see [3] and [4] for the formulae below
    tmp = 1.0
    if n < 50:
        tmp = (
            1.0
            + (2.0 * z - z * z) / (4.0 * n)
            - (24.0 * z - 132.0 * z**2.0 + 76.0 * z**3.0 - 9.0 * z**4.0)
            / (288.0 * n * n)
        )

    p_value = np.exp(-z) * tmp
    return p_value


def h_fpp(H: Union[float, int]) -> float:
    # These values are obtained by fitting to simulations.
    a = 0.9999755
    b = 0.39802
    c = 1.210597
    d = 0.45901
    e = 0.0022900

    if H <= 23:
        return a * np.exp(-b * H)
    elif H < 50:
        return c * np.exp(-d * H + e * H**2)
    else:
        return 4e-8
        # This comes up too often to raise an exception
        raise ValueError(
            f"H={H}>50 not supported; false positive probability less than 4*10**(-8)"
        )


def h_test(events: np.ndarray) -> tuple[float, float, float]:
    """Apply the H test for uniformity on [0,1).

    The H test is an extension of the Z_m^2 or Rayleigh tests for
    uniformity on the circle. These tests estimate the Fourier coefficients
    of the distribution and compare them with the values predicted for
    a uniform distribution, but they require the user to specify the number
    of harmonics to use. The H test automatically selects the number of
    harmonics to use based on the data. The returned statistic, H, has mean
    and standard deviation approximately 2.51, but its significance should
    be evaluated with the routine h_fpp. This is done automatically in this
    routine.

    Arguments
    ---------

    events : array-like
        events should consist of an array of values to be interpreted as
        values modulo 1. These events will be tested for statistically
        significant deviations from uniformity.

    Returns
    -------

    H : float
        The raw score. Larger numbers indicate more non-uniformity.
    M : int
        The number of harmonics that give the most significant deviation
        from uniformity.
    fpp : float
        The probability of an H score this large arising from sampling a
        uniform distribution.

    Reference
    ---------

    de Jager, O. C., Swanepoel, J. W. H, and Raubenheimer, B. C., "A
    powerful test for weak periodic signals of unknown light curve shape
    in sparse data", Astron. Astrophys. 221, 180-190, 1989.
    """
    max_harmonic = 20
    ev = np.reshape(events, (-1,))
    cs = np.sum(
        np.exp(2.0j * np.pi * np.arange(1, max_harmonic + 1) * ev[:, None]), axis=0
    ) / len(ev)
    Zm2 = 2 * len(ev) * np.cumsum(np.abs(cs) ** 2)
    Hcand = Zm2 - 4 * np.arange(1, max_harmonic + 1) + 4
    M = np.argmax(Hcand) + 1
    H = Hcand[M - 1]
    fpp = h_fpp(H)
    return (H, M, fpp)
