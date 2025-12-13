from numba import njit
import numpy as np

from ..types.basic_types import HistStat, BinType


def hist(data: np.ndarray, bins: np.ndarray, stat: HistStat) -> np.ndarray | None:
    if stat == "probability":
        data, _ = np.histogram(data, bins)
        return data / data.sum()
    elif stat == "count":
        data, _ = np.histogram(data, bins)
        return data
    elif stat == "density":
        data, _ = np.histogram(data, bins, density=True)
        return data


@njit(cache=True)
def bin_y_by_x(
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    min_value: float | None,
    max_value: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bins array y by array x. X and Y are expected to be parallel time series data. Where
    x is something like phase and y is something like amplitude.

    Args:
        x (np.ndarray): array
        y (np.ndarray): array
        steps (int): number of bins for binning the data

    Returns:
        _type_: _description_
    """
    if min_value is None:
        min_value = x.min()
    if max_value is None:
        max_value = x.max()
    output_bins = np.linspace(min_value, max_value + 1e-6, num=steps + 1)
    binned_data = np.zeros(steps)
    for i in range(steps):
        subset = y[(x < output_bins[i + 1]) & (x >= output_bins[i])]
        if subset.size > 0:
            binned_data[i] = np.mean(subset)
    return output_bins, binned_data
