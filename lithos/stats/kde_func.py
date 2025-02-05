from typing import Literal, Optional

import KDEpy
import numpy as np
import numpy.typing as npt


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
    tol: float = 1e-3,
    kde_type: Literal["fft", "tree"] = "fft",
):
    if kde_type == "fft":
        data = np.asarray(data)
        kde_obj = KDEpy.FFTKDE(kernel=kernel, bw=bw).fit(data)
        if x is None:
            width = np.sqrt(np.cov(data) * kde_obj.bw**2)
            min_data = data.min() - width * tol
            max_data = data.max() + width * tol
            power2 = int(np.ceil(np.log2(len(data))))
            x = np.linspace(min_data, max_data, num=(1 << power2))
        y = kde_obj.evaluate(x)
    else:
        data = np.asarray(data)
        kde_obj = KDEpy.TreeKDE(kernel=kernel, bw=bw).fit(data)
        if x is None:
            width = np.sqrt(np.cov(data) * kde_obj.bw**2)
            min_data = data.min() - width * tol
            max_data = data.max() + width * tol
            power2 = int(np.ceil(np.log2(len(data))))
            x = np.linspace(min_data, max_data, num=(1 << power2))
        y = kde_obj.evaluate(x)
    return x, y
