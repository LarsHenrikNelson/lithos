import numpy as np

from ..plot_utils import _process_colors, create_dict, process_scatter_args


class BaseProcessor:
    def __init__(self, markers, hatches):
        self.MARKERS = markers
        self.HATCHES = hatches
        self._plot_dict = {}
        self.zorder = 0

    def _create_groupings(self, data, group, subgroup, group_order, subgroup_order):
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

    def _set_zorder(self):
        adder = self.zorder * len(self._plot_dict["zorder_dict"]) + 1
        zorder_dict = {
            key: value + adder for key, value in self._plot_dict["zorder_dict"].items()
        }
        self.zorder += 1
        return zorder_dict

    def preprocess_args(self, args: dict):
        check_args = {"marker", "linestyle"}
        for key, value in args:
            if "color" in key:
                color = _process_colors(
                    value,
                    self._plot_dict["group_order"],
                    self._plot_dict["subgroup_order"],
                )
                args[key] = create_dict(color, self._plot_dict["unique_groups"])
            elif key in check_args:
                args[key] = create_dict(value, self._plot_dict["unique_groups"])
            elif key == "width":
                args[key] = value * self._plot_dict["width"]

    def process_scatter(self, args):
        for key, value in args.items():
            if "color" in key:
                if isinstance(value, tuple):
                    markercolor0 = value[0]
                    markercolor1 = value[1]
                else:
                    markercolor0 = value
                    markercolor1 = None

                args[key] = process_scatter_args(
                    markercolor0,
                    self.data,
                    self._plot_dict["levels"],
                    self._plot_dict["unique_groups"],
                    markercolor1,
                )
            elif key == "markersize":
                if isinstance(key, tuple):
                    column = key[0]
                    start, stop = key[1].split(":")
                    start, stop = int(start) * 4, int(stop) * 4
                    vmin = self.data.min(column)
                    vmax = self.data.max(column)
                    vals = self.data[column]
                    args[key] = (np.array(vals) - vmin) * (stop - start) / (
                        vmax - vmin
                    ) + start
                else:
                    args[key] = [value * 4] * self.data.shape[0]

        args["facetgroup"] = process_scatter_args(
            self._plot_dict["facet"],
            self.data,
            self._plot_dict["levels"],
            self._plot_dict["unique_groups"],
        )

    def __call__(
        self,
        data,
        plot_metadata,
        **kwargs,
    ):
        self.process_groups(data, **plot_metadata["grouping"])

        processed_data = []
        transforms = plot_metadata["transforms"]
        y = plot_metadata["data"]["y"]
        x = plot_metadata["data"]["x"]
        levels = self._plot_dict["levels"]

        for p, pdict in zip(plot_metadata["plot_methods"], plot_metadata["plot_prefs"]):
            if p != "scatter":
                args = self.preprocess_args(pdict)
            else:
                args = self.process_scatter(pdict)
            temp = self.PLOTS[p](
                data=data,
                y=y,
                x=x,
                loc_dict=kwargs["loc_dict"],
                levels=levels,
                **args,
                **transforms,
            )
            processed_data.append(temp)
        return processed_data

    def _process_dict(self, groups, dict, subgroups=None, agg: str | None = None):
        if subgroups is None or agg is not None:
            output = [dict[g] for g in groups.keys()]
        else:
            output = [dict[g[:-1]] for g in subgroups.keys()]
        return output

    # def _biplot(
    #     data: DataHolder,
    #     columns,
    #     group,
    #     subgroup=None,
    #     group_order=None,
    #     subgroup_order=None,
    #     plot_pca=False,
    #     plot_loadings=True,
    #     marker="o",
    #     color="black",
    #     components=None,
    #     alpha=0.8,
    #     labelsize=20,
    #     axis=True,
    # ):
    #     if components is None:
    #         components = (0, 1)
    #     X = preprocessing.scale(data[columns])
    #     pca = decomposition.PCA(n_components=np.max(components) + 1)
    #     X = pca.fit_transform(X)
    #     loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    #     fig, ax = plt.subplots()

    #     if plot_pca:
    #         if group_order is None:
    #             group_order = np.unique(data[group])
    #         if subgroup is None:
    #             subgroup_order = [""]
    #         if subgroup_order is None:
    #             subgroup_order = np.unique(data[subgroup])

    #         unique_groups = []
    #         for i in group_order:
    #             for j in subgroup_order:
    #                 unique_groups.append(i + j)
    #         if subgroup is None:
    #             ug_list = data[group]
    #         else:
    #             ug_list = data[group] + data[subgroup]

    #         marker_dict = process_args(marker, group_order, subgroup_order)
    #         color_dict = process_args(color, group_order, subgroup_order)

    #         if components is None:
    #             components = [0, 1]
    #         xs = X[:, components[0]]
    #         ys = X[:, components[1]]
    #         scalex = 1.0 / (xs.max() - xs.min())
    #         scaley = 1.0 / (ys.max() - ys.min())
    #         for ug in unique_groups:
    #             indexes = np.where(ug_list == ug)[0]
    #             ax.scatter(
    #                 xs[indexes] * scalex,
    #                 ys[indexes] * scaley,
    #                 alpha=alpha,
    #                 marker=marker_dict[ug],
    #                 c=color_dict[ug],
    #             )
    #         ax.legend(
    #             marker: str | dict[str, str],
    #         )
    #     if plot_loadings:
    #         width = -0.005 * np.min(
    #             [np.subtract(*ax.get_xlim()), np.subtract(*ax.get_ylim())]
    #         )
    #         for i in range(loadings.shape[0]):
    #             ax.arrow(
    #                 0,
    #                 0,
    #                 loadings[i, 0],
    #                 loadings[i, 1],
    #                 color="grey",
    #                 alpha=0.5,
    #                 width=width,
    #             )
    #             ax.text(
    #                 loadings[i, 0] * 1.15,
    #                 loadings[i, 1] * 1.15,
    #                 columns[i],
    #                 color="grey",
    #                 ha="center",
    #                 va="center",
    #             )
    #     ax.set_xlim(-1.5, 1.5)
    #     ax.set_ylim(-1.5, 1.5)
    #     ax.tick_params(
    #         axis="both",
    #         which="both",
    #         bottom=False,
    #         left=False,
    #         labelbottom=False,
    #         labelleft=False,
    #     )
    #     ax.set_xlabel(
    #         f"PC{components[0]} ({np.round(pca.explained_variance_ratio_[components[0]] * 100,decimals=2)}%)",
    #         fontsize=labelsize,
    #     )
    #     ax.set_ylabel(
    #         f"PC{components[1]} ({np.round(pca.explained_variance_ratio_[components[1]] * 100,decimals=2)}%)",
    #         fontsize=labelsize,
    #     )
    #     ax.spines["top"].set_visible(axis)
    #     ax.spines["right"].set_visible(axis)
    #     ax.spines["left"].set_visible(axis)
    #     ax.spines["bottom"].set_visible(axis)
