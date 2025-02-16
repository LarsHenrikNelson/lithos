from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from ..utils import (
    DataHolder,
    metadata_utils,
)
from . import processing
from .plot_utils import (
    _process_colors,
    _process_positions,
    create_dict,
    process_args,
    process_scatter_args,
)
from .types import (
    BW,
    Agg,
    AlphaRange,
    BinType,
    CapStyle,
    ColorParameters,
    CountPlotTypes,
    Error,
    KDEType,
    SavePath,
    Transform,
)

PLOTS = {
    "box": processing._box,
    "hist": processing._hist,
    "jitter": processing._jitter,
    "jitteru": processing._jitteru,
    "line": processing._line,
    "poly_hist": processing._poly_hist,
    "summary": processing._summary,
    "summaryu": processing._summaryu,
    "violin": processing._violin,
    "kde": processing._kde,
    "percent": processing._percent,
    "ecdf": processing._ecdf,
    "count": processing._count,
    # "scatter": processing._scatter,
    "aggline": processing._aggline,
}


class BasePlot:
    aggregating_funcs = Agg
    error_funcs = Error
    transform_funcs = Transform

    def __init__(self, data: dict | pd.DataFrame | np.ndarray, inplace: bool = False):
        self.inplace = inplace
        self.plots = []
        self.plot_list = []
        self._plot_methods = []
        self._plot_prefs = []
        self._grouping = {}
        self.data = DataHolder(data)

        self.plot_format = {}
        self._plot_dict = {}
        self._plot_data = {}

        if not self.inplace:
            self.inplace = True
            self.labels()
            self.axis()
            self.axis_format()
            self.figure()
            self.grid_settings()
            self.transform()
            self.inplace = False
        else:
            self.labels()
            self.axis()
            self.axis_format()
            self.figure()
            self.grid_settings()
            self.transform()

    def labels(
        self,
        labelsize: float = 20,
        titlesize: float = 22,
        ticklabel_size: int = 12,
        font: str = "DejaVu Sans",
        fontweight: None | str | float = None,
        title_fontweight: str | float = "regular",
        label_fontweight: str | float = "regular",
        tick_fontweight: str | float = "regular",
        xlabel_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
        ylabel_rotation: Literal["horizontal", "vertical"] | float = "vertical",
        xtick_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
        ytick_rotation: Literal["horizontal", "vertical"] | float = "horizontal",
    ):
        if fontweight is not None:
            title_fontweight = fontweight
            label_fontweight = fontweight
            tick_fontweight = fontweight

        label_props = {
            "labelsize": labelsize,
            "titlesize": titlesize,
            "font": font,
            "ticklabel_size": ticklabel_size,
            "title_fontweight": title_fontweight,
            "label_fontweight": label_fontweight,
            "tick_fontweight": tick_fontweight,
            "xlabel_rotation": xlabel_rotation,
            "ylabel_rotation": ylabel_rotation,
            "xtick_rotation": xtick_rotation,
            "ytick_rotation": ytick_rotation,
        }
        self.plot_format["labels"] = label_props
        if not self.inplace:
            return self

    def axis(
        self,
        ylim: list | None = None,
        xlim: list | None = None,
        yaxis_lim: list | None = None,
        xaxis_lim: list | None = None,
        yscale: Literal["linear", "log", "symlog"] = "linear",
        xscale: Literal["linear", "log", "symlog"] = "linear",
        ydecimals: int = None,
        xdecimals: int = None,
        xformat: Literal["f", "e"] = "f",
        yformat: Literal["f", "e"] = "f",
        yunits: Literal["degree", "radian" "wradian"] | None = None,
        xunits: Literal["degree", "radian" "wradian"] | None = None,
    ):
        if ylim is None:
            ylim = [None, None]
        if xlim is None:
            xlim = [None, None]

        axis_settings = {
            "yscale": yscale,
            "xscale": xscale,
            "ylim": ylim,
            "xlim": xlim,
            "yaxis_lim": yaxis_lim,
            "xaxis_lim": xaxis_lim,
            "ydecimals": ydecimals,
            "xdecimals": xdecimals,
            "xunits": xunits,
            "yunits": yunits,
            "xformat": xformat,
            "yformat": yformat,
        }
        self.plot_format["axis"] = axis_settings

        if not self.inplace:
            return self

    def axis_format(
        self,
        linewidth: float = 2,
        tickwidth: float = 2,
        ticklength: float = 5.0,
        minor_tickwidth: float = 1.5,
        minor_ticklength: float = 2.5,
        yminorticks: bool = False,
        xminorticks: bool = False,
        ysteps: int | tuple[int, int, int] = 5,
        xsteps: int | tuple[int, int, int] = 5,
    ):
        if isinstance(ysteps, int):
            ysteps = (ysteps, 0, ysteps)
        if isinstance(xsteps, int):
            xsteps = (xsteps, 0, xsteps)
        if isinstance(linewidth, int):
            linewidth = {"left": linewidth, "bottom": linewidth, "top": 0, "right": 0}
        elif isinstance(linewidth, dict):
            temp_lw = {"left": 0, "bottom": 0, "top": 0, "right": 0}
            for key, value in linewidth:
                temp_lw[key] = value
            linewidth = temp_lw

        axis_format = {
            "tickwidth": tickwidth,
            "ticklength": ticklength,
            "linewidth": linewidth,
            "minor_tickwidth": minor_tickwidth,
            "minor_ticklength": minor_ticklength,
            "yminorticks": yminorticks,
            "xminorticks": xminorticks,
            "xsteps": xsteps,
            "ysteps": ysteps,
        }

        self.plot_format["axis_format"] = axis_format

        if not self.inplace:
            return self

    def figure(
        self,
        margins=0.05,
        aspect: int | float = 1.0,
        figsize: None | tuple[int, int] = None,
        gridspec_kw: dict[str, str | int | float] = None,
        nrows: int = None,
        ncols: int = None,
        projection: Literal["rectilinear", "polar"] = "rectilinear",
    ):

        figure = {
            "gridspec_kw": gridspec_kw,
            "margins": margins,
            "aspect": aspect if projection == "rectilinear" else None,
            "figsize": figsize,
            "nrows": nrows,
            "ncols": ncols,
            "projection": projection,
        }

        self.plot_format["figure"] = figure

        if not self.inplace:
            return self

    def grid_settings(
        self,
        ygrid: bool = False,
        xgrid: bool = False,
        linestyle: str | tuple = "solid",
        ylinewidth: float | int = 1,
        xlinewidth: float | int = 1,
    ):

        grid_settings = {
            "ygrid": ygrid,
            "xgrid": xgrid,
            "linestyle": linestyle,
            "xlinewidth": xlinewidth,
            "ylinewidth": ylinewidth,
        }
        self.plot_format["grid"] = grid_settings

        if not self.inplace:
            return self

    def clear_plots(self):
        self.plots = []
        self._plot_methods = []
        self._plot_prefs = []

        if not self.inplace:
            return self

    def plot(
        self,
        savefig: bool = False,
        path: SavePath = None,
        filename: str = "",
        filetype: str = "svg",
        backend: str = "matplotlib",
        save_metadata: bool = False,
    ):

        if backend == "matplotlib":
            output = self._matplotlib_backend(
                savefig=savefig, path=path, filetype=filetype, filename=filename
            )
        else:
            raise AttributeError("Backend not implemented")
        if save_metadata:
            path = Path(path)
            filename = self._plot_data["y"] if filename == "" else filename
            path = path / f"{filename}.txt"
            self.save_metadata(path)
        return output

    def transform(
        self,
        ytransform: Transform | None = None,
        back_transform_yticks: bool = False,
        xtransform: Transform | None = None,
        back_transform_xticks: bool = False,
    ):
        self._plot_transforms = {}
        self._plot_transforms["ytransform"] = ytransform
        if callable(ytransform):
            self._plot_transforms["back_transform_yticks"] = False
        else:
            self._plot_transforms["back_transform_yticks"] = back_transform_yticks

        self._plot_transforms["xtransform"] = xtransform
        if callable(xtransform):
            self._plot_transforms["back_transform_xticks"] = False
        else:
            self._plot_transforms["back_transform_xticks"] = back_transform_xticks

        if not self.inplace:
            return self

    def get_format(self):
        return self.plot_format

    def plot_data(
        self,
        y: str | None = None,
        x: str | None = None,
        ylabel: str = "",
        xlabel: str = "",
        title: str = "",
    ):
        if x is None and y is None:
            raise AttributeError("Must specify either x or y")
        self._plot_data = {
            "y": y,
            "x": x,
            "ylabel": ylabel,
            "xlabel": xlabel,
            "title": title,
        }

        if not self.inplace:
            return self

    def _create_groupings(self, group, subgroup, group_order, subgroup_order):
        if group is None:
            unique_groups = [("",)]
            group_order = [""]
            levels = []
        elif subgroup is None:
            if group_order is None:
                group_order = np.unique(self.data[group])
            unique_groups = [(g,) for g in group_order]
            levels = [group]
        else:
            if group_order is None:
                group_order = np.unique(self.data[group])
            if subgroup_order is None:
                subgroup_order = np.unique(self.data[subgroup])
            unique_groups = list(set(zip(self.data[group], self.data[subgroup])))
            levels = [group, subgroup]
        return group_order, subgroup_order, unique_groups, levels

    def metadata(self):
        output = {
            "grouping": self._grouping,
            "data": self._plot_data,
            "format": self.plot_format,
            "transforms": self._plot_transforms,
            "plot_methods": self._plot_methods,
            "plot_prefs": self._plot_prefs,
        }
        return output

    def save_metadata(self, file_path: str | Path):
        metadata = self.metadata()
        metadata_utils.save_metadata(metadata, file_path)

    def _load_plot_prefs(self, plot_dict: dict, meta_dict: dict):
        for key, value in meta_dict.items():
            if key in plot_dict:
                plot_dict[key] = value

    def _set_metadata_from_dict(self, metadata: dict):
        self._plot_data = metadata["data"]
        for key in metadata["format"]:
            self._load_plot_prefs(self.plot_format[key], metadata["format"][key])
        self._load_plot_prefs(self._plot_transforms, metadata["transforms"])

        # Not super happy with this code but it works for now
        if not self.inplace:
            self.inplace = True
            self.grouping(**metadata["grouping"])

            # Must loop through metadata and not set class variables otherwise it will
            # overwrite the instance and get stuck before the loop even runs.
            for pfunc, ppref in zip(metadata["plot_methods"], metadata["plot_prefs"]):
                method = getattr(self, pfunc)
                method(**ppref)
        else:
            self.grouping(**metadata["grouping"])
            # Must loop through metadata and not set class variables otherwise it will
            # overwrite the instance and get stuck before the loop even runs.
            for pfunc, ppref in zip(metadata["plot_methods"], metadata["plot_prefs"]):
                method = getattr(self, pfunc)
                method(**ppref)
        self.inplace = False
        return self

    def load_metadata(self, metadata_path: str | dict | Path):
        metadata = metadata_utils.load_metadata(metadata_path)
        if not self.inplace:
            self = self._set_metadata_from_dict(metadata)
            return self

    def set_metadata_directory(self, metadata_dir: str | dict | Path):
        metadata_utils.set_metadata_dir(metadata_dir)


class LinePlot(BasePlot):
    ecdf_args = {
        "spline": {"size": 1000, "bc_type": "natural"},
        "bootstrap": {"size": 1000, "repititions": 1000, "seed": 42},
    }

    def __init__(self, data: pd.DataFrame, inplace: bool = False):
        super().__init__(data, inplace)

        if not self.inplace:
            self.inplace = True
            self.grouping()
            self.inplace = False
        else:
            self.grouping()

    def grouping(
        self,
        group: str | int | None = None,
        subgroup: str | int | None = None,
        group_order: list[str | int | float] | None = None,
        subgroup_order: list[str | int | float] | None = None,
        facet: bool = False,
        facet_title: bool = False,
    ):
        self._grouping = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "facet": facet,
            "facet_title": facet_title,
        }
        group_order, subgroup_order, unique_groups, levels = self._create_groupings(
            group, subgroup, group_order, subgroup_order
        )

        if facet:
            facet_dict = create_dict(group_order, unique_groups)
        else:
            facet_dict = create_dict(0, unique_groups)

        self._plot_dict = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "facet": facet,
            "facet_dict": facet_dict,
            "facet_title": facet_title,
            "levels": levels,
        }

        if not self.inplace:
            return self

    def line(
        self,
        linecolor: ColorParameters = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
    ):
        self._plot_methods.append("line")
        self._plot_prefs.append(
            {
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "alpha": alpha,
                "unique_id": unique_id,
            }
        )

        linecolor_dict = create_dict(linecolor, self._plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self._plot_dict["unique_groups"])
        line_plot = {
            "color_dict": linecolor_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "alpha": alpha,
            "unique_id": unique_id,
        }
        self.plot_list.append(("line_plot", line_plot))

        if not self.inplace:
            return self

    def aggline(
        self,
        marker: str = "none",
        markerfacecolor: ColorParameters | tuple[str, str] = None,
        markeredgecolor: ColorParameters | tuple[str, str] = None,
        markersize: float | str = 1,
        linecolor: ColorParameters = None,
        linewidth: float = 1.0,
        linestyle: str = "-",
        linealpha: float = 1.0,
        func="mean",
        err_func="sem",
        agg_func=None,
        fill_between: bool = False,
        fillalpha: AlphaRange = 1.0,
        sort=True,
        unique_id=None,
    ):
        self._plot_methods.append("aggline")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markerfacecolor": markerfacecolor,
                "markeredgecolor": markeredgecolor,
                "markersize": markersize,
                "linecolor": linecolor,
                "linewidth": linewidth,
                "linestyle": linestyle,
                "linealpha": linealpha,
                "func": func,
                "err_func": err_func,
                "agg_func": agg_func,
                "fill_between": fill_between,
                "fillalpha": fillalpha,
                "sort": sort,
                "unique_id": None,
            }
        )
        linecolor = _process_colors(
            linecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(
            linecolor,
            self._plot_dict["unique_groups"],
        )
        markerfacecolor_dict = create_dict(
            markerfacecolor,
            self._plot_dict["unique_groups"],
        )
        markeredgecolor_dict = create_dict(
            markeredgecolor,
            self._plot_dict["unique_groups"],
        )

        marker_dict = create_dict(marker, self._plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self._plot_dict["unique_groups"])

        line_plot = {
            "linecolor": linecolor_dict,
            "linestyle": linestyle_dict,
            "linewidth": linewidth,
            "func": func,
            "err_func": err_func,
            "linealpha": linealpha,
            "fill_between": fill_between,
            "fillalpha": fillalpha,
            "sort": sort,
            "marker": marker_dict,
            "markerfacecolor": markerfacecolor_dict,
            "markeredgecolor": markeredgecolor_dict,
            "markersize": markersize,
            "unique_id": unique_id,
            "agg_func": agg_func,
        }
        self.plot_list.append(("aggline", line_plot))

        if not self.inplace:
            return self

    def kde(
        self,
        kernel: KDEType = "gaussian",
        bw: BW = "silverman",
        tol: float | int = 1e-3,
        common_norm: bool = True,
        linecolor: ColorParameters = None,
        linestyle: str = "-",
        linewidth: int = 2,
        fill_between: bool = False,
        alpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 1.0,
        kde_length: int | None = None,
        unique_id: str | None = None,
        agg_func=None,
        err_func=None,
        KDEType: Literal["tree", "fft"] = "fft",
    ):
        self._plot_methods.append("kde")
        self._plot_prefs.append(
            {
                "kernel": kernel,
                "bw": bw,
                "tol": tol,
                "common_norm": common_norm,
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "fill_between": fill_between,
                "alpha": alpha,
                "fillalpha": fillalpha,
                "kde_length": kde_length,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "KDEType": KDEType,
            }
        )

        linecolor = _process_colors(
            linecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linecolor = create_dict(
            linecolor,
            self._plot_dict["unique_groups"],
        )

        linestyle = create_dict(
            linestyle,
            self._plot_dict["unique_groups"],
        )

        kde_plot = {
            "linecolor": linecolor,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "linealpha": alpha,
            "fill_between": fill_between,
            "kernel": kernel,
            "bw": bw,
            "tol": tol,
            "common_norm": common_norm,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "err_func": err_func,
            "kde_length": kde_length,
            "KDEType": KDEType,
            "fillalpha": alpha / 2 if fillalpha is None else fillalpha,
        }

        self.plot_list.append(("kde", kde_plot))

        if not self.inplace:
            return self

    def polyhist(
        self,
        color: ColorParameters = None,
        linestyle: str = "-",
        linewidth: int = 2,
        bin_limits=None,
        density=True,
        nbins=50,
        func="mean",
        err_func="sem",
        fit_func=None,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
    ):
        self._plot_methods.append("polyhist")
        self._plot_pref.append(
            {
                "linestyle": linestyle,
                "linewidth": linewidth,
                "bin_limits": bin_limits,
                "density": density,
                "nbins": nbins,
                "func": func,
                "err_func": err_func,
                "fit_func": fit_func,
                "alpha": alpha,
                "unique_id": unique_id,
            }
        )
        color_dict = process_args(
            _process_colors(
                color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            ),
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linestyle_dict = process_args(
            linestyle, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )

        if bin_limits is not None and len(bin_limits) != 2:
            raise AttributeError("bin_limits must be length 2")

        poly_hist = {
            "color_dict": color_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "density": density,
            "bin_limits": bin_limits,
            "nbins": nbins,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "unique_id": unique_id,
            "alpha": alpha,
        }
        self.plot_list.append(("poly_hist", poly_hist))

        if not self.inplace:
            return self

    def hist(
        self,
        hist_type: Literal["bar", "step", "stepfilled"] = "bar",
        color: ColorParameters = None,
        linecolor: ColorParameters = None,
        linewidth: int = 2,
        hatch=None,
        fillalpha: AlphaRange = 1.0,
        linealpha: float = 1.0,
        bin_limits=None,
        stat: Literal["density", "probability", "count"] = "density",
        nbins=50,
        err_func=None,
        agg_func=None,
        unique_id=None,
    ):
        self._plot_methods.append("hist")
        self._plot_prefs.append(
            {
                "hist_type": hist_type,
                "color": color,
                "linecolor": linecolor,
                "linewidth": linewidth,
                "hatch": hatch,
                "bin_limits": bin_limits,
                "fillalpha": fillalpha,
                "linealpha": linealpha,
                "nbins": nbins,
                "err_func": err_func,
                "agg_func": agg_func,
                "stat": stat,
                "unique_id": unique_id,
            }
        )
        color = _process_colors(
            color,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])
        linecolor = _process_colors(
            linecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(linecolor, self._plot_dict["unique_groups"])

        hist = {
            "color_dict": color_dict,
            "linecolor_dict": linecolor_dict,
            "linewidth": linewidth,
            "hatch": hatch,
            "stat": stat,
            "bin_limits": bin_limits,
            "nbins": nbins,
            "agg_func": agg_func,
            "err_func": err_func,
            "unique_id": unique_id,
            "fillalpha": fillalpha,
            "linealpha": linealpha,
            "projection": self.plot_format["figure"]["projection"],
        }
        self.plot_list.append(("hist", hist))

        if self.plot_format["figure"]["projection"] == "polar":
            self.plot_format["grid"]["ygrid"] = True
            self.plot_format["grid"]["xgrid"] = True

        if not self.inplace:
            return self

    def ecdf(
        self,
        marker: str = "none",
        markerfacecolor: ColorParameters | tuple[str, str] = None,
        markeredgecolor: ColorParameters | tuple[str, str] = None,
        markersize: float | str = 1,
        linecolor: ColorParameters = None,
        linestyle: str = "-",
        linewidth: int = 2,
        linealpha: AlphaRange = 1.0,
        fill_between: bool = True,
        fillalpha: AlphaRange = 0.5,
        unique_id: str | None = None,
        agg_func=None,
        err_func=None,
        colorall: ColorParameters = None,
        ecdf_type: Literal["spline", "bootstrap", "none"] = "none",
        ecdf_args=None,
    ):
        if ecdf_args is None and agg_func is not None:
            ecdf_args = {"size": 1000, "repititions": 1000, "seed": 42}
            ecdf_type = "bootstrap"
        self._plot_methods.append("ecdf")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markerfacecolor": markerfacecolor,
                "markeredgecolor": markeredgecolor,
                "markersize": markersize,
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "linealpha": linealpha,
                "fill_between": fill_between,
                "fillalpha": fillalpha,
                "ecdf_type": ecdf_type,
                "agg_func": agg_func,
                "err_func": err_func,
                "colorall": colorall,
                "ecdf_args": ecdf_args,
            }
        )
        if colorall is None:
            linecolor = _process_colors(
                linecolor,
                self._plot_dict["group_order"],
                self._plot_dict["subgroup_order"],
            )
            linecolor_dict = create_dict(
                linecolor,
                self._plot_dict["unique_groups"],
            )
            markerfacecolor_dict = create_dict(
                markerfacecolor,
                self._plot_dict["unique_groups"],
            )
            markeredgecolor_dict = create_dict(
                markeredgecolor,
                self._plot_dict["unique_groups"],
            )
        else:
            temp_dict = create_dict(colorall, self._plot_dict["unique_groups"])
            markeredgecolor_dict = temp_dict
            markerfacecolor_dict = temp_dict
            linecolor_dict = temp_dict

        marker_dict = create_dict(marker, self._plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self._plot_dict["unique_groups"])

        ecdf = {
            "linecolor": linecolor_dict,
            "linestyle": linestyle_dict,
            "linewidth": linewidth,
            "linealpha": linealpha,
            "marker": marker_dict,
            "markersize": markersize,
            "markerfacecolor": markerfacecolor_dict,
            "markeredgecolor": markeredgecolor_dict,
            "unique_id": unique_id,
            "ecdf_type": ecdf_type,
            "ecdf_args": ecdf_args if ecdf_args is not None else {},
            "agg_func": agg_func,
            "err_func": err_func,
            "fillalpha": fillalpha,
            "fill_between": fill_between,
        }
        self.plot_list.append(("ecdf", ecdf))

        self.plot_format["axis"]["ylim"] = [0.0, 1.0]

        if not self.inplace:
            return self

    def scatter(
        self,
        marker: str = ".",
        markercolor: ColorParameters | tuple[str, str] = "black",
        edgecolor: ColorParameters = "black",
        markersize: float | str = 1,
        alpha: AlphaRange = 1.0,
        line_alpha: AlphaRange = 1.0,
    ):
        self._plot_methods.append("scatter")
        self._plot_prefs.append(
            {
                "marker": marker,
                "markercolor": markercolor,
                "edgecolor": edgecolor,
                "markersize": markersize,
                "alpha": alpha,
                "line_alpha": line_alpha,
            }
        )

        if isinstance(markercolor, tuple):
            markercolor0 = markercolor[0]
            markercolor1 = markercolor[1]
        else:
            markercolor0 = markercolor
            markercolor1 = None

        if isinstance(edgecolor, tuple):
            edgecolor0 = edgecolor[0]
            edgecolor1 = edgecolor[1]
        else:
            edgecolor0 = edgecolor
            edgecolor1 = None

        colors = process_scatter_args(
            markercolor0,
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
            markercolor1,
            alpha=alpha,
        )
        edgecolors = process_scatter_args(
            edgecolor0,
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
            edgecolor1,
            alpha=alpha,
        )
        markersize = process_scatter_args(
            markersize,
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
        )
        facetgroup = process_scatter_args(
            self._plot_dict["facet"],
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
        )
        scatter = {
            "markers": marker,
            "markercolors": colors,
            "edgecolors": edgecolors,
            "markersizes": markersize,
            "facetgroup": facetgroup,
            "alpha": alpha,
            "linealpha": line_alpha
        }

        self.plot_list.append(("scatter", scatter))

        if not self.inplace:
            return self

    def fit(
        self,
        fit_func: Callable,
        linecolor: ColorParameters = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        alpha: AlphaRange = 1.0,
        unique_id: str | None = None,
        agg_func: Agg = None,
        err_func: Error = None,
    ):
        self._plot_methods.append("fit")
        self._plot_prefs.append(
            {
                "linecolor": linecolor,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "alpha": alpha,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "err_func": err_func,
            }
        )


class CategoricalPlot(BasePlot):
    def __init__(self, data: pd.DataFrame | np.ndarray | dict, inplace: bool = False):
        super().__init__(data, inplace)

        if not self.inplace:
            self.inplace = True
            self.grouping()
            self.inplace = False
        else:
            self.grouping()

    def grouping(
        self,
        group: str | int | float = None,
        subgroup: str | int | float = None,
        group_order: list[str | int | float] | None = None,
        subgroup_order: list[str | int | float] | None = None,
        group_spacing: float | int = 1.0,
    ):
        self._grouping = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "group_spacing": group_spacing,
        }
        group_order, subgroup_order, unique_groups, levels = self._create_groupings(
            group, subgroup, group_order, subgroup_order
        )

        if group is not None:
            loc_dict, width = _process_positions(
                subgroup=subgroup,
                group_order=group_order,
                subgroup_order=subgroup_order,
                group_spacing=group_spacing,
            )
        else:
            group_order = [("",)]
            subgroup_order = [("",)]
            loc_dict = {("",): 0.0}
            loc_dict[("",)] = 0.0
            width = 1.0

        x_ticks = [index for index, _ in enumerate(group_order)]
        self._plot_dict = {
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "x_ticks": x_ticks,
            "loc_dict": loc_dict,
            "width": width,
            "levels": levels,
        }

        if not self.inplace:
            return self

    def jitter(
        self,
        color: ColorParameters = None,
        marker: str | dict[str, str] = "o",
        edgecolor: ColorParameters = "none",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: float | int = 1.0,
        seed: int = 42,
        markersize: float = 2.0,
        unique_id: str | None = None,
        legend: bool = False,
    ):
        self._plot_methods.append("jitter")
        self._plot_prefs.append(
            {
                "color": color,
                "marker": marker,
                "edgecolor": edgecolor,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "width": width,
                "markersize": markersize,
                "seed": seed,
                "unique_id": unique_id,
                "legend": legend,
            }
        )
        marker_dict = create_dict(marker, self._plot_dict["unique_groups"])
        color = _process_colors(
            color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])

        if edgecolor == "color":
            edgecolor_dict = color_dict
        else:
            edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        jitter_plot = {
            "color_dict": color_dict,
            "marker_dict": marker_dict,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "width": width * self._plot_dict["width"],
            "seed": seed,
            "markersize": markersize,
            "unique_id": unique_id,
        }
        self.plot_list.append(("jitter", jitter_plot))

        if legend:
            if color is not None or edgecolor == "none":
                d = color
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def jitteru(
        self,
        unique_id: str | int | float,
        color: ColorParameters = None,
        marker: str | dict[str, str] = "o",
        edgecolor: ColorParameters = "none",
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: float | int = 1.0,
        duplicate_offset=0.0,
        markersize: float = 2.0,
        agg_func: Agg | None = None,
        legend: bool = False,
    ):
        self._plot_methods.append("jitteru")
        self._plot_prefs.append(
            {
                "unique_id": unique_id,
                "color": color,
                "marker": marker,
                "edgecolor": edgecolor,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "width": width,
                "duplicate_offset": duplicate_offset,
                "markersize": markersize,
                "agg_func": agg_func,
                "legend": legend,
            }
        )
        marker_dict = create_dict(marker, self._plot_dict["unique_groups"])
        color = _process_colors(
            color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])

        if edgecolor is None:
            edgecolor_dict = color_dict
        else:
            edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        if edge_alpha is None:
            edge_alpha = alpha

        jitteru_plot = {
            "color_dict": color_dict,
            "marker_dict": marker_dict,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "width": width * self._plot_dict["width"],
            "markersize": markersize,
            "unique_id": unique_id,
            "duplicate_offset": duplicate_offset,
            "agg_func": agg_func,
        }
        self.plot_list.append(("jitteru", jitteru_plot))

        if legend:
            if color is not None or edgecolor == "none":
                d = color
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def summary(
        self,
        func: Agg = "mean",
        capsize: int = 0,
        capstyle: CapStyle = "round",
        barwidth: float = 1.0,
        err_func: Error = "sem",
        linewidth: int = 2,
        color: ColorParameters = "black",
        alpha: float = 1.0,
        legend: bool = False,
    ):
        self._plot_methods.append("summary")
        self._plot_prefs.append(
            {
                "func": func,
                "capsize": capsize,
                "capstyle": capstyle,
                "barwidth": barwidth,
                "err_func": err_func,
                "linewidth": linewidth,
                "color": color,
                "alpha": alpha,
                "legend": legend,
            }
        )
        color = _process_colors(
            color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])

        summary_plot = {
            "func": func,
            "capsize": capsize,
            "capstyle": capstyle,
            "barwidth": barwidth * self._plot_dict["width"],
            "err_func": err_func,
            "linewidth": linewidth,
            "color_dict": color_dict,
            "alpha": alpha,
        }

        self.plot_list.append(("summary", summary_plot))

        if legend:
            d = _process_colors(
                color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def summaryu(
        self,
        unique_id,
        func: Agg = "mean",
        agg_func: Agg = None,
        agg_width: float = 1.0,
        capsize: int = 0,
        capstyle: CapStyle = "round",
        barwidth: float = 1.0,
        err_func: Error = "sem",
        linewidth: int = 2,
        color: ColorParameters = "black",
        alpha: float = 1.0,
        legend: bool = False,
    ):
        self._plot_methods.append("summaryu")
        self._plot_prefs.append(
            {
                "func": func,
                "unique_id": unique_id,
                "agg_func": agg_func,
                "agg_width": agg_width,
                "capsize": capsize,
                "capstyle": capstyle,
                "barwidth": barwidth,
                "err_func": err_func,
                "linewidth": linewidth,
                "color": color,
                "alpha": alpha,
                "legend": legend,
            }
        )
        color = _process_colors(
            color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])

        summary_plot = {
            "func": func,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "capsize": capsize,
            "capstyle": capstyle,
            "barwidth": barwidth * self._plot_dict["width"],
            "err_func": err_func,
            "linewidth": linewidth,
            "color_dict": color_dict,
            "alpha": alpha,
            "agg_width": agg_width,
        }

        self.plot_list.append(("summaryu", summary_plot))

        if legend:
            d = _process_colors(
                color, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def box(
        self,
        facecolor: ColorParameters = None,
        edgecolor: ColorParameters = None,
        fliers="",
        width: float = 1.0,
        linewidth=1,
        alpha: AlphaRange = 1.0,
        linealpha: AlphaRange = 1.0,
        showmeans: bool = False,
        show_ci: bool = False,
        legend: bool = False,
    ):
        self._plot_methods.append("box")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "fliers": fliers,
                "width": width,
                "alpha": alpha,
                "linewidth": linewidth,
                "linealpha": linealpha,
                "showmeans": showmeans,
                "show_ci": show_ci,
                "legend": legend,
            }
        )
        if facecolor != "edgecolor":
            color = _process_colors(
                facecolor,
                self._plot_dict["group_order"],
                self._plot_dict["subgroup_order"],
            )
            color_dict = create_dict(color, self._plot_dict["unique_groups"])

        if edgecolor != "facecolor":
            edgecolor = _process_colors(
                edgecolor,
                self._plot_dict["group_order"],
                self._plot_dict["subgroup_order"],
            )
            edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        if facecolor == "edgecolor":
            color_dict = edgecolor_dict

        if edgecolor == "facecolor":
            edgecolor_dict = color_dict

        box = {
            "color_dict": color_dict,
            "edgecolor_dict": edgecolor_dict,
            "fliers": fliers,
            "width": width * self._plot_dict["width"],
            "showmeans": showmeans,
            "show_ci": show_ci,
            "linewidth": linewidth,
            "alpha": alpha,
            "linealpha": linealpha,
        }

        self.plot_list.append(("box", box))

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def violin(
        self,
        facecolor: ColorParameters = None,
        edgecolor: ColorParameters = None,
        linewidth=1,
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = 1.0,
        showextrema: bool = False,
        width: float = 1.0,
        showmeans: bool = False,
        showmedians: bool = False,
        legend: bool = False,
    ):
        self._plot_methods.append("violin")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": linewidth,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "showextrema": showextrema,
                "width": width,
                "showmeans": showmeans,
                "showmedians": showmedians,
                "legend": legend,
            }
        )
        color = _process_colors(
            facecolor, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self._plot_dict["unique_groups"])
        edgecolor = _process_colors(
            edgecolor, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        edge_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])
        violin = {
            "facecolor_dict": color_dict,
            "edgecolor_dict": edge_dict,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "showextrema": showextrema,
            "width": width * self._plot_dict["width"],
            "showmeans": showmeans,
            "showmedians": showmedians,
            "linewidth": linewidth,
        }

        self.plot_list.append(("violin", violin))

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def percent(
        self,
        cutoff: None | float | int | list[float | int],
        unique_id=None,
        facecolor=None,
        edgecolor: ColorParameters = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        linealpha=1.0,
        axis_type: BinType = "density",
        include_bins: list[bool] | None = None,
        invert: bool = False,
        legend: bool = False,
    ):
        self._plot_methods.append("percent")
        self._plot_prefs.append(
            {
                "cutoff": cutoff,
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "linewidth": linewidth,
                "barwidth": barwidth,
                "alpha": alpha,
                "linealpha": linealpha,
                "axis_type": axis_type,
                "invert": invert,
                "include_bins": include_bins,
                "legend": legend,
            }
        )
        if isinstance(cutoff, (float, int)):
            cutoff = [cutoff]
        facecolor = _process_colors(
            facecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        color_dict = create_dict(facecolor, self._plot_dict["unique_groups"])

        edgecolor = _process_colors(
            edgecolor,
            self._plot_dict["group_order"],
            self._plot_dict["subgroup_order"],
        )
        edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        percent_plot = {
            "color_dict": color_dict,
            "edgecolor_dict": edgecolor_dict,
            "cutoff": cutoff,
            "hatch": hatch,
            "barwidth": barwidth * self._plot_dict["width"],
            "linewidth": linewidth,
            "alpha": alpha,
            "linealpha": linealpha,
            "include_bins": include_bins,
            "unique_id": unique_id,
            "invert": invert,
            "axis_type": axis_type,
        }

        self.plot_list.append(("percent", percent_plot))

        if axis_type == "density":
            self.plot_format["axis"]["ylim"] = [0.0, 1.0]
        else:
            self.plot_format["axis"]["ylim"] = [0, 100]

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def count(
        self,
        facecolor: ColorParameters = None,
        edgecolor: ColorParameters = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        edge_alpha=1.0,
        axis_type: CountPlotTypes = "count",
        legend: bool = False,
    ):
        self._plot_methods.append("count")
        self._plot_prefs.append(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "barwidth": barwidth,
                "linewidth": linewidth,
                "alpha": alpha,
                "edge_alpha": edge_alpha,
                "axis_type": axis_type,
                "legend": legend,
            }
        )
        facecolor = _process_colors(
            facecolor, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        color_dict = create_dict(facecolor, self._plot_dict["unique_groups"])

        edgecolor = _process_colors(
            edgecolor, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
        )
        edgecolor_dict = create_dict(edgecolor, self._plot_dict["unique_groups"])

        count_plot = {
            "color_dict": color_dict,
            "edgecolor_dict": edgecolor_dict,
            "hatch": hatch,
            "barwidth": barwidth * self._plot_dict["width"],
            "linewidth": linewidth,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "axis_type": axis_type,
        }
        self.plot_list.append(("count", count_plot))

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self._plot_dict["group_order"], self._plot_dict["subgroup_order"]
            )
            self._plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self


class GraphPlot:

    def __init__(self, graph):
        self._plot_dict = {}
        self._plot_dict["graph"] = graph
        self.plots = []

    def graphplot(
        self,
        marker_alpha: float = 0.8,
        linealpha: float = 0.1,
        markersize: int = 2,
        markerscale: int = 1,
        linewidth: int = 1,
        edgecolor: str = "k",
        markercolor: str = "red",
        marker_attr: str | None = None,
        cmap: str = "gray",
        seed: int = 42,
        scale: int = 50,
        plot_max_degree: bool = False,
        layout: Literal["spring", "circular", "communities"] = "spring",
    ):
        graph_plot = {
            "marker_alpha": marker_alpha,
            "linealpha": linealpha,
            "markersize": markersize,
            "markerscale": markerscale,
            "linewidth": linewidth,
            "edgecolor": edgecolor,
            "markercolor": markercolor,
            "marker_attr": marker_attr,
            "cmap": cmap,
            "seed": seed,
            "scale": scale,
            "layout": layout,
            "plot_max_degree": plot_max_degree,
        }
        self.plots.append(graph_plot)
