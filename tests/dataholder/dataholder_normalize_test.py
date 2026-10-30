import numpy as np
import pandas as pd
import pytest

from lithos.utils import DataHolder


class TestDataHolderNormalize:
    def test_dict_input_is_not_mutated(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        DataHolder(data)
        assert all(isinstance(value, list) for value in data.values())

    def test_dataframe_input_is_reused_without_copy(self):
        frame = pd.DataFrame({"a": [1, 2, 3]})
        assert DataHolder(frame).to_pd() is frame

    def test_dataholder_input_shares_frame(self):
        holder = DataHolder({"a": [1, 2, 3]})
        assert DataHolder(holder).to_pd() is holder.to_pd()

    def test_series_input_becomes_single_column_frame(self):
        holder = DataHolder(pd.Series([1.0, 2.0, 3.0], name="y"))
        assert holder.shape == (3, 1)
        assert list(holder[np.arange(2), "y"]) == [1.0, 2.0]

    def test_1d_array_becomes_single_column(self):
        holder = DataHolder(np.arange(4.0))
        assert holder.shape == (4, 1)
        assert list(holder[0]) == [0.0, 1.0, 2.0, 3.0]

    def test_2d_array_columns_are_positional_ints(self):
        holder = DataHolder(np.arange(6.0).reshape(3, 2))
        assert 1 in holder
        assert list(holder[:, 1]) == [1.0, 3.0, 5.0]

    def test_3d_array_raises(self):
        with pytest.raises(ValueError):
            DataHolder(np.zeros((2, 2, 2)))

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            DataHolder({})

    def test_ragged_dict_raises_with_clear_message(self):
        with pytest.raises(ValueError, match="same length"):
            DataHolder({"a": [1, 2, 3], "b": [1, 2]})

    def test_unsupported_input_raises(self):
        with pytest.raises(TypeError):
            DataHolder([1, 2, 3])

    def test_len_is_row_count(self):
        holder = DataHolder({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert len(holder) == 3
        assert len(holder) == holder.shape[0]


class TestDataHolderContains:
    def test_numpy_origin_string_is_false_not_error(self):
        holder = DataHolder(np.arange(6.0).reshape(3, 2))
        assert "a" not in holder

    def test_numpy_origin_out_of_range_values_are_false(self):
        holder = DataHolder(np.arange(6.0).reshape(3, 2))
        assert -5 not in holder
        assert 2 not in holder
        assert 1.5 not in holder


class TestDataHolderGetItem:
    def test_missing_column_raises_key_error(self):
        holder = DataHolder(np.arange(6.0).reshape(3, 2))
        with pytest.raises(KeyError):
            holder["a"]

    def test_invalid_single_index_raises_type_error(self):
        holder = DataHolder({"a": [1, 2, 3]})
        with pytest.raises(TypeError):
            holder[np.arange(2)]

    def test_tuple_with_wrong_length_raises(self):
        holder = DataHolder({"a": [1, 2, 3]})
        with pytest.raises(TypeError):
            holder[(np.arange(2),)]

    def test_list_rows_tuple_indexing(self):
        holder = DataHolder(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        assert list(holder[[0, 2], "a"]) == [1, 3]

    def test_bool_mask_rows_tuple_indexing(self):
        holder = DataHolder({"a": [1, 2, 3]})
        mask = np.array([True, False, True])
        assert list(holder[mask, "a"]) == [1, 3]

    def test_multiple_columns_return_dataframe(self):
        holder = DataHolder({"g": [1, 1, 2], "a": [1, 2, 3], "b": [4, 5, 6]})
        out = holder[np.array([0, 2]), ["a", "b"]]
        assert isinstance(out, pd.DataFrame)
        assert out.to_numpy().tolist() == [[1, 4], [3, 6]]

    def test_missing_tuple_column_raises_with_names(self):
        holder = DataHolder({"a": [1, 2, 3]})
        with pytest.raises(KeyError, match="not found"):
            holder[np.arange(2), ["a", "zzz"]]


class TestDataHolderGroupby:
    def test_sort_false_is_respected(self):
        holder = DataHolder({"g": [2, 1, 2, 1], "y": [1, 2, 3, 4]})
        sorted_keys = list(holder.groupby("y", ("g",), sort=True).indices.keys())
        unsorted_keys = list(holder.groupby("y", ("g",), sort=False).indices.keys())
        assert sorted_keys == [(1,), (2,)]
        assert unsorted_keys == [(2,), (1,)]

    def test_returns_fresh_groupby_objects(self):
        holder = DataHolder({"g": [1, 2], "y": [1, 2]})
        assert holder.groupby("y", ("g",)) is not holder.groupby("y", ("g",))
