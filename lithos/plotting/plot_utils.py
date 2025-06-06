from fractions import Fraction

import colorcet as cc
import matplotlib as mpl
import numpy as np
from numpy.random import default_rng

from .types import Group, Subgroup, UniqueGroups


def _calc_hist(data, bins, stat):
    if stat == "probability":
        data, _ = np.histogram(data, bins)
        return data / data.sum()
    elif stat == "count":
        data, _ = np.histogram(data, bins)
        return data
    elif stat == "density":
        data, _ = np.histogram(data, bins, density=True)
        return data


def create_dict(grouping: str | int | dict, unique_groups: list) -> dict:
    """
    Create a dictionary based on the grouping and unique groups.
    The function assumes that the group and subgroup passed to the plot class
    do not have any of the same elements in common.

    Args:
        grouping (str | int | dict): _description_
        unique_groups (list): _description_

    Returns:
        dict: Output dictionary
    """
    if grouping is None or isinstance(grouping, (str, int)):
        output_dict = {key: grouping for key in unique_groups}
    else:
        if not isinstance(grouping, dict):
            grouping = {key: value for value, key in enumerate(grouping)}
        output_dict = {}
        for i in grouping.keys():
            for j in unique_groups:
                if isinstance(i, tuple) and isinstance(j, tuple):
                    if len(i) != len(j):
                        if i == j[: len(i)]:
                            output_dict[j] = grouping[i]
                    elif i == j:
                        output_dict[j] = grouping[i]
                elif i in j:
                    output_dict[j] = grouping[i]
    return output_dict


def _get_colormap(colormap: str | None):
    if colormap is None:
        colormap = "glasbey_category10"
    if colormap in cc.palette:
        return cc.palette[colormap]
    elif colormap in mpl.colormaps:
        if isinstance(mpl.colormaps[colormap], mpl.colors.LinearSegmentedColormap):
            return [mpl.colormaps[colormap](i) for i in np.linspace(0, 1, 256)]
        else:
            return mpl.colormaps[colormap].colors
    else:
        raise ValueError(
            f"Colormap '{colormap}' not found in colorcet or matplotlib colormaps."
        )


def _process_colormap(color: str, groups: list):
    color = color.split("-")
    if color[0] in cc.palette or color[0] in mpl.colormaps:
        color_palette = _get_colormap(color[0])
    else:
        raise ValueError("Colormap not recognized")
    if len(color) == 2:
        one, two = color[1].split(":")
        one = max(0, int(one))
        two = min(255, int(two))
        num = len(groups)
        indexes = np.linspace(one, two, num=num, dtype=int)
    else:
        indexes = np.arange(len(groups))
    output_color = {key: color_palette[index] for key, index in zip(groups, indexes)}
    return output_color


def _process_string_color(color: str, group_order: list, subgroup_order: list):
    if color in mpl.colors.CSS4_COLORS:
        return color
    elif color in mpl.colors.BASE_COLORS:
        return color
    elif color in mpl.colors.TABLEAU_COLORS:
        return color
    elif color in mpl.colors.XKCD_COLORS:
        return mpl.colors.XKCD_COLORS[color]
    elif color == "none":
        return color
    elif color is None or color in cc.palette or color in mpl.colormaps or ":" in color:
        group = subgroup_order if subgroup_order is not None else group_order
        color = _process_colormap(color, group)
        return color
    else:
        raise ValueError("Color not recognized, must be str, 'none', or None")


def _process_colors(
    color: str | list | dict | None,
    group_order: list | None = None,
    subgroup_order: list | None = None,
):
    """
    This function prepocesses the color parameter so that the color specified by the
    user is compatible with the create_dict function. If the color is a str that is not
    a recognized colormap or dictionary than the function just returns that same object.
    If a colormap or None is specified than the function creates a list of the colormap.
    Any list objects are then processing in loop that assigns to colors to subgroup_order,
    group_order or just takes the first items in the list if either subgroup_order or
    group_order are None.

    Args:
        color (str | list | dict | None): _description_
        group_order (list | None, optional): _description_. Defaults to None.
        subgroup_order (list | None, optional): _description_. Defaults to None.

    Returns:
        str | dict: Color output that can be a string or dictionary
    """
    if isinstance(color, str):
        color = _process_string_color(
            color, group_order=group_order, subgroup_order=subgroup_order
        )
        return color
    if isinstance(color, dict):
        return color
    if isinstance(color, Group):
        return {key: value for key, value in zip(group_order, color.group)}
    if isinstance(color, Subgroup):
        return {key: value for key, value in zip(subgroup_order, color.subgroup)}
    if isinstance(color, UniqueGroups):
        output = {}
        index = 0
        for g in group_order:
            for s in subgroup_order:
                output[(g, s)] = color.unique_groups[index]
                index += 1
        return output


def radian_ticks(ticks, rotate=False):
    pi_symbol = "\u03c0"
    mm = [int(180 * i / np.pi) for i in ticks]
    if rotate:
        mm = [deg if deg <= 180 else deg - 360 for deg in mm]
    jj = [Fraction(deg / 180) if deg != 0 else 0 for deg in mm]
    output = []
    for t in jj:
        sign = "-" if t < 0 else ""
        t = abs(t)
        if t.numerator == 0 or t == 0:
            output.append("0")
        elif t.numerator == 1 and t.denominator == 1:
            output.append(f"{sign}{pi_symbol}")
        elif abs(t.denominator) == 1:
            output.append(f"{t.numerator}{pi_symbol}")
        elif abs(t.numerator) == 1:
            output.append(f"{sign}{pi_symbol}/{t.denominator}")
        else:
            output.append(f"{sign}{t.numerator}{pi_symbol}/{t.denominator}")
    return output


def process_duplicates(values, output=None):
    vals, counts = np.unique(
        values,
        return_counts=True,
    )
    track_counts = {}
    if output is None:
        output = np.zeros(values.size)
    for key, val in zip(vals, counts):
        if val > 1:
            track_counts[key] = [0, np.linspace(-1, 1, num=val)]
        else:
            track_counts[key] = [0, [0]]
    for index, val in enumerate(values):
        output[index] += track_counts[val][1][track_counts[val][0]]
        track_counts[val][0] += 1
    return output


def process_jitter(values, loc, width, rng=None, seed=42):
    if rng is None:
        rng = default_rng(seed)
    try:
        counts, _ = np.histogram(values, bins="doane")
    except Exception:
        counts, _ = np.histogram(values, bins="sqrt")
    jitter_values = np.zeros(len(values))
    asort = np.argsort(values)
    start = 0
    s = (-width / 2) + loc
    e = (width / 2) + loc
    for c in counts:
        if c != 0:
            if c == 1:
                temp = rng.random(size=1)
                temp *= width / 4
                temp -= width / 4
                temp += loc
            else:
                temp = rng.permutation(np.linspace(s, e, num=c))
            jitter_values[asort[start : start + c]] = temp
            start += c
    return jitter_values


def _invert(array, invert):
    if invert:
        if isinstance(array, list):
            array.reverse()
        else:
            array = array[::-1]
        return array
    else:
        return array


def get_ticks(
    lim,
    axis_lim,
    ticks,
    steps,
):
    if lim[0] is None:
        bottom = ticks[0]
    else:
        bottom = lim[0]
    if lim[1] is None:
        top = ticks[-1]
    else:
        top = lim[1]
    lim = (bottom, top)
    if axis_lim is None:
        axis_lim = (bottom, top)
    if axis_lim[0] is None:
        axis_bottom = lim[0]
    else:
        axis_bottom = axis_lim[0]
    if axis_lim[1] is None:
        axis_top = lim[1]
    else:
        axis_top = axis_lim[1]
    axis_lim = (axis_bottom, axis_top)
    ticks = np.linspace(
        axis_lim[0],
        axis_lim[1],
        steps[0],
    )
    return lim, axis_lim, ticks


def _bin_data(data, bins, axis_type, invert, cutoff):
    if cutoff is not None:
        temp = np.sort(data)
        binned_data, _ = np.histogram(temp, bins)
    else:
        binned_data = np.zeros(len(bins))
        conv_dict = {key: value for value, key in enumerate(bins)}
        unames, ucounts = np.unique(data, return_counts=True)
        for un, uc in zip(unames, ucounts):
            binned_data[conv_dict[un]] = uc
    binned_data = binned_data / binned_data.sum()
    if axis_type == "percent":
        binned_data *= 100
    if invert:
        binned_data = binned_data[::-1]
    bottom = np.zeros(len(binned_data))
    bottom[1:] = binned_data[:-1]
    bottom = np.cumsum(bottom)
    top = binned_data
    return top, bottom


def _decimals(data):
    temp = np.abs(data)
    temp = temp[temp > 0.0]
    decimals = np.abs(int(np.max(np.round(np.log10(temp))))) + 2
    return decimals


def _process_groups(df, group, subgroup, group_order, subgroup_order):
    if group is None:
        return ["none"], [""]
    if group_order is None:
        group_order = sorted(df[group].unique())
    else:
        if len(group_order) != len(df[group].unique()):
            raise AttributeError(
                "The number groups does not match the number in group_order"
            )
    if subgroup is not None:
        if subgroup_order is None:
            subgroup_order = sorted(df[subgroup].unique())
        elif len(subgroup_order) != len(df[subgroup].unique()):
            raise AttributeError(
                "The number subgroups does not match the number in subgroup_order"
            )
    else:
        subgroup_order = [""] * len(group_order)
    return group_order, subgroup_order


def bin_data(data, bins):
    binned_data = np.zeros(bins.size - 1, dtype=int)
    index = 0
    for i in data:
        if index >= bins.size:
            binned_data[binned_data.size - 1] += 1
        elif i >= bins[index] and i < bins[int(index + 1)]:
            binned_data[index] += 1
        else:
            if index < binned_data.size - 1:
                index += 1
                binned_data[index] += 1
            elif index < binned_data.size:
                binned_data[index] += 1
                index += 1
            else:
                binned_data[binned_data.size - 1] += 1
    return binned_data


def process_args(arg, group, subgroup):
    if isinstance(arg, (str, int, float)):
        arg = {key: arg for key in group}
    elif isinstance(arg, list):
        arg = {key: arg for key, arg in zip(group, arg)}
    output_dict = {}
    for s in group:
        for b in subgroup:
            key = rf"{s}" + rf"{b}"
            if s in arg:
                output_dict[key] = arg[s]
            else:
                output_dict[key] = arg[b]
    return output_dict


def process_scatter_args(
    arg, data, levels, group_order, subgroup_order, unique_groups, arg_cycle=None
):
    if isinstance(arg, dict):
        output = create_dict(arg, unique_groups)
        if len(levels) > 0:
            output = [output[j] for j in zip(*[data[i] for i in levels])]
        else:
            output = [0] * data.shape[0]
        return output
    if isinstance(arg_cycle, (np.ndarray, list)):
        if arg in data:
            if arg_cycle is not None:
                output = _discrete_cycler(arg, data, arg_cycle)
            else:
                output = data[arg]
        elif len(arg) < len(unique_groups):
            output = arg
    elif isinstance(arg_cycle, str):
        if ":" in arg_cycle:
            arg_cycle, indexes = arg_cycle.split("-")
            start, stop = indexes.split(":")
        else:
            start, stop = 0, 255
    if arg_cycle in cc.palette or arg_cycle in mpl.colormaps:
        if arg not in data:
            raise AttributeError("arg[0] of arg must be in data passed to LinePlot")
        output = _continuous_cycler(arg, data, arg_cycle, start, stop)
    else:
        if arg in cc.palette or arg_cycle in mpl.colormaps:
            arg = _process_colors(arg, group_order, subgroup_order)
        output = create_dict(arg, unique_groups)
        if len(levels) > 0:
            output = [output[j] for j in zip(*[data[i] for i in levels])]
        else:
            output = [output[("",)]] * data.shape[0]
    return output


def _discrete_cycler(arg, data, arg_cycle):
    grps = np.unique(data[arg])
    ntimes = data.shape[0] // len(arg_cycle)
    markers = arg_cycle
    if ntimes > 0:
        markers = markers * (ntimes + 1)
        markers = markers[: data.shape[0]]
    mapping = {key: value for key, value in zip(grps, markers)}
    output = [mapping(key) for key in data[arg]]
    return output


def _continuous_cycler(arg, data, arg_cycle, start=0, stop=255):
    cmap = _get_colormap(arg_cycle)
    start = max(0, int(start))
    stop = min(len(cmap), int(stop))
    uvals = set(data[arg])
    if len(uvals) != len(data[arg]):
        vmax = len(uvals)
        cvals = np.linspace(start, stop - 1, num=vmax, dtype=int)
        mapping = {key: cmap[c] for c, key in zip(cvals, uvals)}
        colors = [mapping[key] for key in data[arg]]
    else:
        vmin = min(data[arg])
        vmax = max(data[arg])
        vals = data[arg]
        color_normal = (np.array(vals) - vmin) * ((stop - 1) - start) / (
            vmax - vmin
        ) + start
        color_normal = color_normal.astype(int)
        colors = [cmap[e] for e in color_normal]
    return colors


def _process_positions(group_spacing, group_order, subgroup_order=None):
    group_loc = {key: float(index) for index, key in enumerate(group_order)}
    if subgroup_order is not None:
        width = group_spacing / len(subgroup_order)
        start = (group_spacing / 2) - (width / 2)
        sub_loc = np.linspace(-start, start, len(subgroup_order))
        subgroup_loc = {key: value for key, value in zip(subgroup_order, sub_loc)}
        loc_dict = {}
        for i, i_value in group_loc.items():
            for j, j_value in subgroup_loc.items():
                key = (i, j)
                loc_dict[key] = float(i_value + j_value)

    else:
        loc_dict = {(key,): value for key, value in group_loc.items()}
        width = 1.0
    return loc_dict, width


def _create_groupings(data, group, subgroup, group_order, subgroup_order):
    if group is None:
        unique_groups = [("",)]
        group_order = [""]
        levels = ()
    elif subgroup is None:
        if group_order is None:
            group_order = np.unique(data[group])
        unique_groups = [(g,) for g in group_order]
        levels = (group,)
    else:
        if group_order is None:
            group_order = np.unique(data[group])
        if subgroup_order is None:
            subgroup_order = np.unique(data[subgroup])
        unique_groups = list(set(zip(data[group], data[subgroup])))
        levels = (group, subgroup)
    return group_order, subgroup_order, unique_groups, levels
