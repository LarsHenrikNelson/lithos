from typing import Literal

import numpy as np
from numpy.random import default_rng


def timeseries_dict(loc, scale, n):
    rng = default_rng()
    x1 = np.linspace(0, 50, n)
    x2 = np.linspace(0, 25, n)
    y = np.sin(x1) + rng.random(n) + np.cos(x2)
    y += loc
    y *= scale
    return y


def get_dist(distribution, rng):
    if distribution == "normal":
        return lambda scale, loc, n: np.sort(rng.normal(loc, scale, n))
    elif distribution == "gamma":
        return lambda scale, loc, n: np.sort(rng.gamma(loc, scale, n))
    elif distribution == "timeseries":
        return timeseries_dict
    elif distribution == "lognormal":
        return lambda scale, loc, n: np.sort(rng.lognormal(loc, scale, n))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def create_synthetic_data(
    n_groups: int = 1,
    n_subgroups: int = 0,
    n_unique_ids: int = 0,
    n_points: int = 30,
    seed: int = 42,
    distribution: Literal["normal", "lognormal", "gamma", "timeseries"] = "normal",
    loc: float = 1.2,
    scale: float = 1.0,
):
    unique_grouping = None if n_unique_ids == 0 else []

    rng = default_rng(seed)

    dist = get_dist(distribution, rng)

    additive = []

    groups = np.arange(n_groups)
    if n_subgroups != 0:
        subgroups = np.arange(n_subgroups) + groups.max() + 1
    unique_groups = []
    for i in groups:
        if n_subgroups != 0:
            for j in subgroups:
                unique_groups.append((i, j))
            value = rng.integers(0, 3, size=1)
            additive.extend([value] * len(subgroups))
        else:
            unique_groups.append((i,))
            additive.append(rng.integers(0, 3, size=1))
    grouping_1 = []
    x_data = []
    y_data = []
    if n_subgroups != 0:
        grouping_2 = []
    for index, i in enumerate(unique_groups):
        if n_unique_ids == 0:
            y_data.extend(dist(loc, scale, n_points) + additive[index])
            x_data.extend(np.arange(n_points))
            grouping_1.extend([i[0]] * n_points)
            if n_subgroups != 0:
                grouping_2.extend([i[1]] * n_points)
        else:
            unique_vals = rng.normal(loc, 0.2, n_unique_ids)
            for j in range(n_unique_ids):
                y_data.extend(dist(unique_vals[j], scale, n_points) + additive[index])
                x_data.extend(np.arange(n_points))
                unique_grouping.extend([j] * n_points)
                grouping_1.extend([i[0]] * n_points)
                if n_subgroups != 0:
                    grouping_2.extend([i[1]] * n_points)

    output = {"y": y_data, "x": x_data, "grouping_1": grouping_1}
    if n_subgroups != 0:
        output["grouping_2"] = grouping_2
    if n_unique_ids != 0:
        output["unique_grouping"] = unique_grouping
    return output
