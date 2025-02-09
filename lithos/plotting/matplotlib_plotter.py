from typing import Literal

import matplotlib.patches as mpatches
from matplotlib._enums import CapStyle
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt


class MPLPlotter:
    def __init__(self):
        pass

    def _make_legend_patches(self, color_dict, alpha, group, subgroup):
        legend_patches = []
        # for j in group:
        #     if j in color_dict:
        #         legend_patches.append(
        #             mpatches.Patch(color=to_rgba(color_dict[j], alpha=alpha), label=j)
        #         )
        # for j in subgroup:
        #     if j in color_dict:
        #         legend_patches.append(
        #             mpatches.Patch(color=to_rgba(color_dict[j], alpha=alpha), label=j)
        #         )
        for key, value in color_dict.items():
            legend_patches.append(
                mpatches.Patch(color=to_rgba(value, alpha=alpha), label=key)
            )
        return legend_patches

    def _add_rectangles(
        self,
        tops: list,
        bottoms: list,
        bins: list,
        binwidths: list,
        fillcolors: list[str],
        edgecolors: list[str],
        fill_alpha: float,
        edge_alpha: float,
        hatches: list[str],
        linewidth: float,
        ax: plt.Axes,
        axis: Literal["x", "y"] = "x",
    ):
        for t, b, b, bw, fc, ec, ht, ln, sub_ax in zip(
            tops,
            bottoms,
            bins,
            binwidths,
            fillcolors,
            edgecolors,
            hatches,
            linewidth,
            ax,
        ):
            if axis == "x":
                sub_ax.bar(
                    x=bins,
                    height=t,
                    bottom=b,
                    width=bw,
                    color=to_rgba(fc, alpha=fill_alpha),
                    edgecolor=to_rgba(ec, edge_alpha),
                    linewidth=linewidth,
                    hatch=ht,
                )
            else:
                sub_ax.barh(
                    y=bins,
                    width=t,
                    left=b,
                    height=bw,
                    color=to_rgba(fc, alpha=fill_alpha),
                    edgecolor=to_rgba(ec, edge_alpha),
                    linewidth=linewidth,
                    hatch=hatches,
                )
        return ax

    def _plot_scatter(
        self,
        x_data: list,
        y_data: list,
        marker: str,
        markerfacecolor: list[str],
        markeredgecolor: list[str],
        markersize: float,
        alpha: float,
        edge_alpha: float,
        ax: plt.Axes,
    ):
        for x, y, mk, mf, me, ms in zip(
            x_data, y_data, marker, markerfacecolor, markeredgecolor, markersize
        ):
            ax.plot(
                x,
                y,
                mk,
                markerfacecolor=to_rgba(mf, alpha=alpha) if mf != "none" else "none",
                markeredgecolor=(
                    to_rgba(me, alpha=edge_alpha) if me != "none" else "none"
                ),
                markersize=ms,
            )
        return ax

    def _plot_summary(
        self,
        x_data: list,
        y_data: list,
        error_data: list,
        widths: list,
        colors: list,
        linewidth: float,
        alpha: float,
        capstyle: str,
        capsize: float,
        ax: plt.Axes,
    ):
        for xd, yd, e, c, w in zip(x_data, y_data, error_data, colors, widths):
            _, caplines, bars = ax.errorbar(
                x=xd,
                y=yd,
                yerr=e,
                c=to_rgba(c, alpha=alpha),
                fmt="none",
                linewidth=linewidth,
                capsize=capsize,
            )
            for cap in caplines:
                cap.set_solid_capstyle(capstyle)
                cap.set_markeredgewidth(linewidth)
                cap._marker._capstyle = CapStyle(capstyle)
            for b in bars:
                b.set_capstyle(capstyle)
            _, caplines, bars = ax.errorbar(
                y=yd,
                x=xd,
                xerr=w / 2,
                c=to_rgba(c, alpha=alpha),
                fmt="none",
                linewidth=linewidth,
                capsize=0,
            )
            for b in bars:
                b.set_capstyle(capstyle)
        return ax

    def _plot_boxplot(
        self,
        x_data: list,
        y_data: list,
        facecolors: list[str],
        edgecolors: list[str],
        alpha: float,
        line_alpha: float,
        fliers: bool,
        linewidth: float,
        width: float,
        show_ci: bool,
        showmeans: bool,
        ax: plt.Axes,
    ):
        for x, y, fcs, ecs in zip(x_data, y_data, facecolors, edgecolors):
            props = {
                "boxprops": {
                    "facecolor": to_rgba(fcs, alpha=alpha) if fcs != "none" else fcs,
                    "edgecolor": (
                        to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                    ),
                },
                "medianprops": {
                    "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                },
                "whiskerprops": {
                    "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                },
                "capprops": {
                    "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                },
            }
            if showmeans:
                props["meanprops"] = {
                    "color": to_rgba(ecs, alpha=line_alpha) if ecs != "none" else ecs
                }
            bplot = ax.boxplot(
                y,
                positions=x,
                sym=fliers,
                widths=width,
                notch=show_ci,
                patch_artist=True,
                showmeans=showmeans,
                meanline=showmeans,
                **props,
            )
            for i in bplot["boxes"]:
                i.set_linewidth(linewidth)

    def _plot_violin(
        self,
        x_data: list,
        y_data: list,
        facecolors: list[str],
        edgecolors: list[str],
        alpha: float,
        edge_alpha: float,
        linewidth: float,
        width: list[float],
        showmeans: bool,
        showmedians: bool,
        showextrema: bool,
        ax: plt.Axes,
    ):
        for x, y, fcs, ecs in zip(x_data, y_data, facecolors, edgecolors):
            parts = ax.violinplot(
                y,
                positions=x,
                widths=width,
                showmeans=showmeans,
                showmedians=showmedians,
                showextrema=showextrema,
            )
            for body in parts["bodies"]:
                body.set_facecolor(to_rgba(fcs, alpha=alpha) if fcs != "none" else fcs)
                body.set_edgecolor(
                    to_rgba(ecs, alpha=edge_alpha) if ecs != "none" else ecs
                )
                body.set_linewidth(linewidth)
            if showmeans:
                # Matplotlib seems to divide the alpha by 2 so this needs to be equivalent
                parts["cmeans"].set_color(
                    to_rgba(ecs, alpha=edge_alpha / 2) if ecs != "none" else ecs
                )
                parts["cmeans"].set_linewidth(linewidth)
            if showmedians:
                parts["cmedians"].set_color(
                    to_rgba(ecs, alpha=edge_alpha / 2) if ecs != "none" else ecs
                )
                parts["cmedians"].set_linewidth(linewidth)

    def _plot_error_line(
        self,
        ax: plt.Axes,
        x_data: list,
        y_data: list,
        error_data: list,
        facet_index: list[int],
        marker: list[str | None] | None = None,
        linecolor: list[str | None] | None = None,
        linewidth: list[float | None] | None = None,
        linestyle: list[str | None] | None = None,
        markerfacecolor: list[str | None] | None = None,
        markeredgecolor: list[str | None] | None = None,
        fill_between: bool = False,
        fb_direction: Literal["x", "y"] = "y",
        markersize: float | None = None,
        fillalpha: float | None = None,
        linealpha: float | None = None,
    ):
        for x, y, err, ls, lc, mf, me, mk, fc in zip(
            x_data,
            y_data,
            error_data,
            linestyle,
            linecolor,
            markerfacecolor,
            markeredgecolor,
            marker,
            facet_index,
        ):
            if not fill_between:
                if fb_direction == "x":
                    ax[fc].errorbar(
                        x,
                        y,
                        xerr=err,
                        marker=mk,
                        color=lc,
                        elinewidth=linewidth,
                        linewidth=linewidth,
                        linestyle=ls,
                        markerfacecolor=mf,
                        markeredgecolor=me,
                        markersize=markersize,
                        alpha=linealpha,
                    )
                else:
                    ax[fc].errorbar(
                        x,
                        y,
                        yerr=err,
                        marker=mk,
                        color=lc,
                        elinewidth=linewidth,
                        linewidth=linewidth,
                        linestyle=ls,
                        markerfacecolor=mf,
                        markeredgecolor=me,
                        markersize=markersize,
                        alpha=linealpha,
                    )
            else:
                if err is not None:
                    if fb_direction == "y":
                        ax[fc].fill_between(
                            x,
                            y - err,
                            y + err,
                            color=to_rgba(lc, alpha=fillalpha),
                            linewidth=0,
                            edgecolor="none",
                        )
                    else:
                        ax[fc].fill_betweenx(
                            y,
                            x - err,
                            x + err,
                            color=to_rgba(lc, alpha=fillalpha),
                            linewidth=0,
                            edgecolor="none",
                        )
                ax[fc].plot(
                    x,
                    y,
                    linestyle=ls,
                    linewidth=linewidth,
                    color=lc,
                    alpha=linealpha,
                )
