import numpy as np
import pandas as pd


class DataHolder:

    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = np.array(value)
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
        self._data = data._data if isinstance(data, DataHolder) else data
        self._container_type = self._get_container_type()
        self._groups_cache = {}
        self._groupby_cache = {}

    def __contains__(self, item):
        if self._container_type == "pandas" or self._container_type == "dict":
            return item in self._data
        elif self._container_type == "numpy":
            return item < self._data.shape[1]

    def _get_container_type(self):
        if isinstance(self._data, (pd.Series, pd.DataFrame)):
            return "pandas"
        elif isinstance(self._data, np.ndarray):
            return "numpy"
        elif isinstance(self._data, dict):
            return "dict"
        else:
            raise ValueError(
                "Only numpy arrays, dict, or pandas dataframes/series are accepted."
            )

    def _numpy_index(self, index):
        if isinstance(index, tuple):
            return self._data[index[0], index[1]]
        elif isinstance(index, (list, np.ndarray)):
            return zip([self.data[i] for i in index])
        elif isinstance(index, int):
            return self._data[:, index]

    def _pandas_index(self, index):
        if isinstance(index, tuple):
            if pd.api.types.is_bool_dtype(index[0].dtype) or isinstance(
                index[0], pd.Index
            ):
                return self._data.loc[index[0], index[1]]
            else:
                return self._data.iloc[index[0], self._data.columns.get_loc(index[1])]
        elif isinstance(index, (list, np.ndarray)):
            return zip([self.data[i] for i in index])
        elif isinstance(index, str):
            return self._data[index]

    def _dict_index(self, index):
        if isinstance(index, tuple):
            return self._data[index[1]][index[0]]
        elif isinstance(index, (list, np.ndarray)):
            return zip([self.data[i] for i in index])
        elif isinstance(index, str):
            return self._data[index]
        else:
            raise ValueError("Invalid index type")

    def __getitem__(self, index):
        if self._container_type == "numpy":
            return self._numpy_index(index)
        elif self._container_type == "pandas":
            return self._pandas_index(index)
        elif self._container_type == "dict":
            return self._dict_index(index)

    def min(self, index):
        return np.min(self.__getitem__(index))

    def max(self, index):
        return np.max(self.__getitem__(index))

    @property
    def size(self):
        if self._container_type == "numpy" or self._container_type == "pandas":
            return self._data.size
        elif self._container_type == "dict":
            return len(next(iter(self._data.values()))) * len(self._data.keys())

    @property
    def shape(self):
        if self._container_type == "numpy" or self._container_type == "pandas":
            return self._data.shape
        else:
            return (len(next(iter(self._data.values()))), len(self._data.keys()))

    def __len__(self):
        return self.size

    def get_data(self, levels, ids):
        bool_array = np.full(self.shape[0], True)
        for i, j in zip(levels, ids):
            bool_array = bool_array & (self[i] == j)
        return bool_array

    def groupby(self, y, columns, sort=True):
        if not isinstance(y, tuple):
            y = (y,)
        levels = columns + y
        if levels in self._groupby_cache:
            return self._groupby_cache[levels]
        yy = pd.DataFrame(self._data)[list(columns + y)].groupby(
            list(columns), sort=sort, as_index=False
        )
        self._groupby_cache[levels] = yy
        return yy

    def groups(self, levels):
        if levels in self._groups_cache:
            return self._groups_cache[levels]
        if len(levels) == 0:
            new_groups = {}
            new_groups[("",)] = np.arange(self.shape[0])
            self._groups_cache[levels] = new_groups
            return new_groups
        temp_groups = pd.DataFrame(self._data).groupby(list(levels)).indices
        new_groups = {}
        for key, value in temp_groups.items():
            if not isinstance(key, tuple):
                new_groups[(key,)] = value
            else:
                new_groups[key] = value
        self._groups_cache[levels] = new_groups
        return new_groups
