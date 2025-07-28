from typing import Literal, Optional

import KDEpy
import numpy as np
import numpy.typing as npt


def _kde_length(data, kde_obj, tol: float | int | tuple = 0.1, kde_length: int | None = None):
    if isinstance(tol, tuple):
        min_data, max_data = tol
    else:
        width = np.sqrt(np.cov(data) * kde_obj.bw**2)
        min_data = data.min() - width * tol
        max_data = data.max() + width * tol
    if kde_length is None:
        kde_length = 1 << int(np.ceil(np.log2(len(data))))
    x = np.linspace(min_data, max_data, num=kde_length)
    return x


def kde(
    data: npt.ArrayLike,
    kernel: Literal[
        "gaussian",
        "exponential",
        "box",
        "tri",
        "epa",
        "biweight",
        "triweight",
        "tricube",
        "cosine",
    ] = "gaussian",
    bw: Literal["ISJ", "silverman", "scott"] = "ISJ",
    x: Optional[np.array] = None,
    kde_length: int | None = None,
    tol: float | int | tuple = 1e-3,
    KDEType: Literal["fft", "tree"] = "fft",
):
    data = np.asarray(data)
    if KDEType == "fft":
        kde_obj = KDEpy.FFTKDE(kernel=kernel, bw=bw).fit(data)
    else:
        kde_obj = KDEpy.TreeKDE(kernel=kernel, bw=bw).fit(data)
    if x is None:
        x = _kde_length(data, kde_obj, tol, kde_length)
    y = kde_obj.evaluate(x)
    return x, y
