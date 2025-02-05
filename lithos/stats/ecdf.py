from typing import Literal

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy import interpolate


def ecdf(x, ecdf_type: Literal["bootstrap", "spline", "none"] = "none", **kwargs):
    if ecdf_type == "none":
        x = np.sort(x)
        y = np.arange(x.size) / x.size
    elif ecdf_type == "spline":
        y, x = spline_ecdf(x, **kwargs)
    elif ecdf_type == "bootstrap":
        x = rand_samp_column(x.reshape(1, x.size), **kwargs)
        x = x.flatten()
        y = np.arange(x.size) / x.size
    else:
        raise ValueError("ecdf_type not recognized, must be bootstrap, spline or none")
    return x, y


def spline_ecdf(
    x,
    size,
    bc_type: Literal[
        "not-a-knot",
        "clamped",
        "natural",
    ] = "not-a-knot",
):
    j_sort = np.sort(x)
    y = np.arange(j_sort.size) / j_sort.size
    cs = interpolate.CubicSpline(y, j_sort, bc_type=bc_type)
    y_new = np.arange(size) / size
    x_new = cs(y_new)
    return y_new, x_new


# Could potentially speed this up with numba however,
# numba does not seem to support the new numpy random
# number generator choice method
def rand_samp_column(
    array: np.ndarray,
    repititions: int = 10000,
    size: int = 1000,
    axis: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """Randomly sample from from a row or column in a numpy array with
    replacement.

    Args:
        array (np.array): numpy array with rows or columns to be resampled.
        repititions (int): Number of times to resample each column.
        size (int): Number of samples to take with replacement.
        axis (int): Axis to sample from.

    Returns:
        pd.DataFrame: Resampled data output as a single
    """
    # This extracts "size" random values from each cell x times
    # final_arrays = []
    rng = default_rng(seed)
    if axis == 1:
        array = array.T
    temp_array = np.zeros(array.shape[0] * size)
    extracted_df1 = np.zeros(array.shape[0] * size)
    for i in range(0, repititions):
        index = 0
        for j in range(0, array.shape[0]):
            # Copy column to numpy array
            x1 = array[j, :]

            # Drop nans
            x2 = x1[~np.isnan(x1)]
            if size is None:
                size_1 = len(x2)
            else:
                size_1 = size

            # Choose samples
            b1 = rng.choice(x2, size=size_1)
            extracted_df1[index : index + size] = b1
            index += size
        extracted_df1 = np.sort(extracted_df1)
        temp_array += extracted_df1
    final_array = temp_array / repititions
    return final_array


def sample_from_dfs(
    dfs: dict[str, pd.DataFrame],
    repititions: int,
    size: int,
    axis: int = 1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    sampled_dfs = {}
    for key, df in dfs.items():
        array = rand_samp_column(df.to_numpy(), repititions, size, axis=axis, seed=seed)
        sampled_dfs[key] = pd.DataFrame(array)
    return sampled_dfs


def resampled_cum_prob_df(
    df_dict: dict[str, pd.DataFrame],
    data_id: str,
    group: str,
    repititions: int = 1000,
    size: int = 100,
    axis: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    sampled_dfs = sample_from_dfs(df_dict, repititions, size, axis=axis, seed=seed)
    df_cumsum = []
    for key, df in sampled_dfs.items():
        df.loc[:, data_id] = df.mean(axis=1)
        df.loc[:, "Cumulative Probability"] = (
            1.0 * np.arange(len(df[data_id])) / (len(df[data_id]) - 1)
        )
        df.loc[:, group] = key
        y = df[[data_id, "Cumulative Probability", group]]
        df_cumsum += [y]
    finished_df = pd.concat(df_cumsum)
    return finished_df


def cum_prob_df(dfs, df_keys, data, group):
    """


    Parameters
    ----------
    dfs : TYPE
        DESCRIPTION.
    df_keys : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    group : TYPE
        DESCRIPTION.

    Returns
    -------
    finished_df : TYPE
        DESCRIPTION.

    """
    cum_df_list = []
    for df, key in zip(dfs, df_keys):
        df_stacked = df.stack().reset_index(drop=True).sort_values()
        cum_stacked = 1.0 * np.arange(len(df_stacked)) / (len(df_stacked) - 1)
        cum_df = pd.DataFrame(
            {data: df_stacked.to_numpy(), "Cumulative Probability": cum_stacked}
        )
        cum_df[group] = key
        cum_df_list += [cum_df]
    finished_df = pd.concat(cum_df_list)
    return finished_df
