import itertools
from collections import defaultdict, namedtuple
from typing import Iterable, Sized

import pandas as pd

try:
    from dask import dataframe as dd
except ImportError:  # pragma: no cover
    dd = None


def _is_polars_type(obj, type_name):
    obj_type = type(obj)
    return obj_type.__name__ == type_name and obj_type.__module__.startswith("polars")


def _is_polars_dataframe(obj):
    return _is_polars_type(obj, "DataFrame")


def _is_polars_lazyframe(obj):
    return _is_polars_type(obj, "LazyFrame")


def _is_polars_series(obj):
    return _is_polars_type(obj, "Series")


def _has_nan_values(series):
    if _is_polars_series(series):
        if series.null_count() > 0:
            return True
        try:
            return bool(series.is_nan().any())
        except Exception:  # pragma: no cover
            return False
    return bool(series.isnull().any())


def _sorted_group_items(groups):
    group_items = list(groups.items())
    try:
        return sorted(group_items, key=lambda item: item[0])
    except TypeError:
        return group_items


def _binding_helper(f, kwargs, column_sort, column_id, column_kind, column_value):
    def wrapped_feature_extraction(x):
        # Capture group identity BEFORE sort_values(), because sort_values()
        # returns a new DataFrame that loses the .name attribute set by groupby.apply().
        # In Pandas 3.0+, grouping keys are no longer included in the chunk columns,
        # so we rely on x.name (a tuple like (id_val, kind_val)) to recover them.
        group_name = x.name  # may be None for ungrouped DataFrames

        if column_sort is not None:
            x = x.sort_values(column_sort)

        if column_id in x.columns:
            id_val = x[column_id].iloc[0]
        elif group_name is not None:
            id_val = group_name[0] if isinstance(group_name, tuple) else group_name
        else:
            raise KeyError(
                f"Column '{column_id}' not found in DataFrame chunk and group name unavailable."
            )

        if column_kind in x.columns:
            kind_val = x[column_kind].iloc[0]
        elif group_name is not None:
            kind_val = group_name[1] if isinstance(group_name, tuple) else group_name
        else:
            raise KeyError(
                f"Column '{column_kind}' not found in DataFrame chunk and group name unavailable."
            )

        chunk = Timeseries(id_val, kind_val, x[column_value])
        result = f(chunk, **kwargs)

        result = pd.DataFrame(result, columns=[column_id, "variable", "value"])
        result["value"] = result["value"].astype("double")

        return result[[column_id, "variable", "value"]]

    return wrapped_feature_extraction


class Timeseries(namedtuple("Timeseries", ["id", "kind", "data"])):
    """
    Timeseries tuple used for feature extraction.

    Make sure `kind` is of type `str` to allow inference
    of feature settings in `feature_extraction.settings.from_columns`.
    """


class TsData:
    """
    TsData provides access to time series data for internal usage.

    Distributors will use this data class to apply functions on the data.
    All derived classes must either implement the `apply` method,
    which is used to apply the given function directly on the data
    or the __iter__ method, which can be used to get an iterator of
    Timeseries instances (which distributors can use to apply the function on).
    Other methods can be overwritten if a more efficient solution exists for the underlying data store.
    """

    pass


class PartitionedTsData(Iterable[Timeseries], Sized, TsData):
    """
    Special class of TsData, which can be partitioned.
    Derived classes should implement __iter__ and __len__.
    """

    def __init__(self, df, column_id):
        self.df_id_type = df[column_id].dtype

    def pivot(self, results):
        """
        Helper function to turn an iterable of tuples with three entries into a dataframe.

        The input ``list_of_tuples`` needs to be an iterable with tuples containing three
        entries: (a, b, c).
        Out of this, a pandas dataframe will be created with all a's as index,
        all b's as columns and all c's as values.

        It basically does a pd.pivot(first entry, second entry, third entry),
        but optimized for non-pandas input (= python list of tuples).

        This function is called in the end of the extract_features call.
        """
        return_df_dict = defaultdict(dict)
        for chunk_id, variable, value in results:
            # we turn it into a nested mapping `column -> index -> value`
            return_df_dict[variable][chunk_id] = value

        # the mapping column -> {index -> value}
        # is now a dict of dicts. The pandas dataframe
        # constructor will peel this off:
        # first, the keys of the outer dict (the column)
        # will turn into a column header and the rest into a column
        # the rest is {index -> value} which will be turned into a
        # column with index.
        # All index will be aligned.
        return_df = pd.DataFrame(return_df_dict, dtype=float)

        # copy the type of the index
        return_df.index = return_df.index.astype(self.df_id_type)

        # Sort by index to be backward compatible
        return_df = return_df.sort_index()

        return return_df


def _check_colname(*columns):
    """
    Check if given column names conflict with `settings.from_columns` (ends with '_' or contains '__').

    :param columns: the column names to check
    :type columns: str

    :return: None
    :rtype: None
    :raise: ``ValueError`` if column names are invalid.
    """

    for col in columns:
        if str(col).endswith("_"):
            raise ValueError(
                "Dict keys are not allowed to end with '_': {}".format(col)
            )

        if "__" in str(col):
            raise ValueError(
                "Dict keys are not allowed to contain '__': {}".format(col)
            )


def _check_nan(df, *columns):
    """
    Raise a ``ValueError`` if one of the columns does not exist or contains NaNs.

    :param df: the DataFrame to test for NaNs
    :type df: pandas.DataFrame or polars.DataFrame
    :param columns: a list of columns to test for NaNs. If left empty, all columns of the DataFrame will be tested.
    :type columns: str

    :return: None
    :rtype: None
    :raise: ``ValueError`` if ``NaNs`` are found in the DataFrame.
    """

    for col in columns:
        if col not in df.columns:
            raise ValueError("Column not found: {}".format(col))

        if _has_nan_values(df[col]):
            raise ValueError("Column must not contain NaN values: {}".format(col))


def _get_value_columns(df, *other_columns):
    value_columns = [col for col in df.columns if col not in other_columns]

    if len(value_columns) == 0:
        raise ValueError(
            "Could not guess the value column! Please hand it to the function as an argument."
        )

    return value_columns


class WideTsFrameAdapter(PartitionedTsData):
    def __init__(self, df, column_id, column_sort=None, value_columns=None):
        """
        Adapter for Pandas DataFrames in wide format, where multiple columns contain different time series for
        the same id.

        :param df: the data frame
        :type df: pd.DataFrame

        :param column_id: the name of the column containing time series group ids
        :type column_id: str

        :param column_sort: the name of the column to sort on
        :type column_sort: str|None

        :param value_columns: list of column names to treat as time series values.
            If `None` or empty, all columns except `column_id` and `column_sort` will be used.
        :type value_columns: list[str]|None
        """
        if column_id is None:
            raise ValueError("A value for column_id needs to be supplied")

        _check_nan(df, column_id)

        if not value_columns:
            value_columns = _get_value_columns(df, column_id, column_sort)

        _check_nan(df, *value_columns)
        _check_colname(*value_columns)

        self.value_columns = value_columns

        if column_sort is not None:
            _check_nan(df, column_sort)

        self.column_sort = column_sort
        self.df_grouped = df.groupby(column_id)

        super().__init__(df, column_id)

    def __len__(self):
        return self.df_grouped.ngroups * len(self.value_columns)

    def __iter__(self):
        for group_name, group in self.df_grouped:
            if self.column_sort is not None:
                group = group.sort_values(self.column_sort)

            for kind in self.value_columns:
                yield Timeseries(group_name, kind, group[kind])


class LongTsFrameAdapter(PartitionedTsData):
    def __init__(self, df, column_id, column_kind, column_value=None, column_sort=None):
        """
        Adapter for Pandas DataFrames in long format, where different time series for the same id are
        labeled by column `column_kind`.

        :param df: the data frame
        :type df: pd.DataFrame

        :param column_id: the name of the column containing time series group ids
        :type column_id: str

        :param column_kind: the name of the column containing time series kinds for each id
        :type column_kind: str

        :param column_value: None or the name of the column containing time series values
            If `None`, try to guess it from the remaining, unused columns.
        :type column_value: str|None

        :param column_sort: the name of the column to sort on
        :type column_sort: str|None
        """
        if column_id is None:
            raise ValueError("A value for column_id needs to be supplied")
        if column_kind is None:
            raise ValueError("A value for column_kind needs to be supplied")

        if column_value is None:
            possible_value_columns = _get_value_columns(
                df, column_id, column_sort, column_kind
            )
            if len(possible_value_columns) != 1:
                raise ValueError(
                    "Could not guess the value column, as the number of unused columns os not equal to 1."
                    f"These columns where currently unused: {','.join(possible_value_columns)}"
                    "Please hand it to the function as an argument."
                )
            self.column_value = possible_value_columns[0]
        else:
            self.column_value = column_value

        _check_nan(df, column_id, column_kind, self.column_value)

        if column_sort is not None:
            _check_nan(df, column_sort)

        self.column_sort = column_sort
        self.df_grouped = df.groupby([column_id, column_kind])

        super().__init__(df, column_id)

    def __len__(self):
        return len(self.df_grouped)

    def __iter__(self):
        for group_key, group in self.df_grouped:
            if self.column_sort is not None:
                group = group.sort_values(self.column_sort)
            yield Timeseries(group_key[0], str(group_key[1]), group[self.column_value])


class TsDictAdapter(PartitionedTsData):
    def __init__(self, ts_dict, column_id, column_value, column_sort=None):
        """
        Adapter for a dict, which maps different time series kinds to Pandas DataFrames.

        :param ts_dict: a dict of data frames
        :type ts_dict: dict[str, pd.DataFrame]

        :param column_id: the name of the column containing time series group ids
        :type column_id: str

        :param column_value: the name of the column containing time series values
        :type column_value: str

        :param column_sort: the name of the column to sort on
        :type column_sort: str|None
        """
        _check_colname(*list(ts_dict.keys()))
        for df in ts_dict.values():
            _check_nan(df, column_id, column_value)

        self.column_value = column_value

        if column_sort is not None:
            for key, df in ts_dict.items():
                _check_nan(df, column_sort)

            self.grouped_dict = {
                key: df.sort_values([column_sort]).groupby(column_id)
                for key, df in ts_dict.items()
            }
        else:
            self.grouped_dict = {
                key: df.groupby(column_id) for key, df in ts_dict.items()
            }

        super().__init__(df, column_id)

    def __iter__(self):
        for kind, grouped_df in self.grouped_dict.items():
            for ts_id, group in grouped_df:
                yield Timeseries(ts_id, str(kind), group[self.column_value])

    def __len__(self):
        return sum(grouped_df.ngroups for grouped_df in self.grouped_dict.values())


class PolarsWideTsFrameAdapter(PartitionedTsData):
    def __init__(self, df, column_id, column_sort=None, value_columns=None):
        if column_id is None:
            raise ValueError("A value for column_id needs to be supplied")

        _check_nan(df, column_id)

        if not value_columns:
            value_columns = _get_value_columns(df, column_id, column_sort)

        _check_nan(df, *value_columns)
        _check_colname(*value_columns)

        if column_sort is not None:
            _check_nan(df, column_sort)

        self.value_columns = value_columns
        self.column_sort = column_sort
        self.row_number_column = "_tsfresh_row_number"
        self.df_grouped = df.with_row_count(self.row_number_column).partition_by(
            column_id, maintain_order=True, as_dict=True
        )

        super().__init__(df, column_id)

    def __len__(self):
        return len(self.df_grouped) * len(self.value_columns)

    def __iter__(self):
        for group_name, group in _sorted_group_items(self.df_grouped):
            if self.column_sort is not None:
                group = group.sort([self.column_sort, self.row_number_column])

            if isinstance(group_name, tuple):
                group_name = group_name[0]

            group_index = group.get_column(self.row_number_column).to_list()
            for kind in self.value_columns:
                yield Timeseries(
                    group_name,
                    kind,
                    pd.Series(
                        group.get_column(kind).to_list(), index=group_index, name=kind
                    ),
                )


class PolarsLongTsFrameAdapter(PartitionedTsData):
    def __init__(self, df, column_id, column_kind, column_value=None, column_sort=None):
        if column_id is None:
            raise ValueError("A value for column_id needs to be supplied")
        if column_kind is None:
            raise ValueError("A value for column_kind needs to be supplied")

        if column_value is None:
            possible_value_columns = _get_value_columns(
                df, column_id, column_sort, column_kind
            )
            if len(possible_value_columns) != 1:
                raise ValueError(
                    "Could not guess the value column, as the number of unused columns os not equal to 1."
                    f"These columns where currently unused: {','.join(possible_value_columns)}"
                    "Please hand it to the function as an argument."
                )
            self.column_value = possible_value_columns[0]
        else:
            self.column_value = column_value

        _check_nan(df, column_id, column_kind, self.column_value)

        if column_sort is not None:
            _check_nan(df, column_sort)

        self.column_sort = column_sort
        self.row_number_column = "_tsfresh_row_number"
        self.df_grouped = df.with_row_count(self.row_number_column).partition_by(
            [column_id, column_kind], maintain_order=True, as_dict=True
        )

        super().__init__(df, column_id)

    def __len__(self):
        return len(self.df_grouped)

    def __iter__(self):
        for group_key, group in _sorted_group_items(self.df_grouped):
            if self.column_sort is not None:
                group = group.sort([self.column_sort, self.row_number_column])

            group_index = group.get_column(self.row_number_column).to_list()
            yield Timeseries(
                group_key[0],
                str(group_key[1]),
                pd.Series(
                    group.get_column(self.column_value).to_list(),
                    index=group_index,
                    name=self.column_value,
                ),
            )


class PolarsTsDictAdapter(PartitionedTsData):
    def __init__(self, ts_dict, column_id, column_value, column_sort=None):
        _check_colname(*list(ts_dict.keys()))

        for df in ts_dict.values():
            _check_nan(df, column_id, column_value)

        if column_sort is not None:
            for df in ts_dict.values():
                _check_nan(df, column_sort)

        self.column_value = column_value
        self.column_sort = column_sort
        self.row_number_column = "_tsfresh_row_number"
        self.grouped_dict = {
            key: df.with_row_count(self.row_number_column).partition_by(
                column_id, maintain_order=True, as_dict=True
            )
            for key, df in ts_dict.items()
        }

        super().__init__(next(iter(ts_dict.values())), column_id)

    def __iter__(self):
        for kind, grouped_df in self.grouped_dict.items():
            for ts_id, group in _sorted_group_items(grouped_df):
                if self.column_sort is not None:
                    group = group.sort([self.column_sort, self.row_number_column])

                if isinstance(ts_id, tuple):
                    ts_id = ts_id[0]

                yield Timeseries(
                    ts_id,
                    str(kind),
                    pd.Series(
                        group.get_column(self.column_value).to_list(),
                        index=group.get_column(self.row_number_column).to_list(),
                        name=self.column_value,
                    ),
                )

    def __len__(self):
        return sum(len(grouped_df) for grouped_df in self.grouped_dict.values())


class DaskTsAdapter(TsData):
    def __init__(
        self, df, column_id, column_kind=None, column_value=None, column_sort=None
    ):
        if column_id is None:
            raise ValueError("column_id must be set")

        if column_id not in df.columns:
            raise ValueError(f"Column not found: {column_id}")

        # Get all columns, which are not id, kind or sort
        possible_value_columns = _get_value_columns(
            df, column_id, column_sort, column_kind
        )

        # The user has already a kind column. That means we just need to group by id (and additionally by id)
        if column_kind is not None:
            if column_kind not in df.columns:
                raise ValueError(f"Column not found: {column_kind}")

            self.df = df.groupby([column_id, column_kind])

            # We assume the last remaining column is the value - but there needs to be one!
            if column_value is None:
                if len(possible_value_columns) != 1:
                    raise ValueError(
                        "Could not guess the value column! Please hand it to the function as an argument."
                    )
                column_value = possible_value_columns[0]
        else:
            # Ok, the user has no kind, so it is in Wide format.
            # That means we have do melt before we can group.
            # TODO: here is some room for optimization!
            # we could choose the same way as for the Wide and LongTsAdapter

            # We first choose a name for our future kind column
            column_kind = "kind"

            # if the user has specified a value column, use it
            # if not, just go with every remaining columns
            if column_value is not None:
                value_vars = [column_value]
            else:
                value_vars = possible_value_columns
                column_value = "value"

            # Make sure we are not reusing a column that already exists
            while column_value in df.columns:
                column_value += "_"

            _check_colname(*value_vars)

            id_vars = [column_id, column_sort] if column_sort else [column_id]

            # Now melt and group
            df_melted = df.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=column_kind,
                value_name=column_value,
            )

            self.df = df_melted.groupby([column_id, column_kind])

        self.column_id = column_id
        self.column_kind = column_kind
        self.column_value = column_value
        self.column_sort = column_sort

    def apply(self, f, meta, **kwargs):
        """
        Apply the wrapped feature extraction function "f"
        onto the data.
        Before that, turn the data into the correct form of Timeseries instances
        usable the the feature extraction.
        After the call, turn it back into pandas dataframes
        for further processing.
        """
        bound_function = _binding_helper(
            f,
            kwargs,
            self.column_sort,
            self.column_id,
            self.column_kind,
            self.column_value,
        )
        return self.df.apply(bound_function, meta=meta)

    def pivot(self, results):
        """
        The extract features function for dask returns a
        dataframe of [id, variable, value].
        Turn this into a pivoted dataframe, where only the variables are the columns
        and the ids are the rows.

        Attention: this is highly non-optimized!
        """
        results = results.reset_index(drop=True).persist()
        results = results.categorize(columns=["variable"])
        feature_table = results.pivot_table(
            index=self.column_id, columns="variable", values="value", aggfunc="sum"
        )

        return feature_table


def to_tsdata(
    df, column_id=None, column_kind=None, column_value=None, column_sort=None
):
    """
    Wrap supported data formats as a TsData object, i.e. an iterable of individual time series.

    E.g. the DataFrame

        ====  ======  =========
          id  kind          val
        ====  ======  =========
           1  a       -0.21761
           1  a       -0.613667
           1  a       -2.07339
           2  b       -0.576254
           2  b       -1.21924
        ====  ======  =========

    into

        Iterable((1, 'a', pd.Series([-0.217610, -0.613667, -2.073386]),
         (2, 'b', pd.Series([-0.576254, -1.219238]))

    :param df: one of the supported input formats
    :type df: pd.DataFrame|polars.DataFrame|dict|TsData

    :param column_id: The name of the id column to group by.
    :type column_id: str|None

    :param column_kind: The name of the column keeping record on the kind of the value.
    :type column_kind: str|None

    :param column_value: The name for the column keeping the value itself.
    :type column_value: str|None

    :param column_sort: The name for the column to sort on.
    :type column_sort: str|None

    :return: a data adapter
    :rtype: TsData
    """
    if isinstance(df, TsData):
        return df

    elif _is_polars_lazyframe(df):
        raise ValueError(
            "df must be an eager polars.DataFrame. Polars LazyFrame is not supported. "
            "Please call collect() before passing the data to tsfresh."
        )

    elif isinstance(df, pd.DataFrame):
        if column_kind is not None:
            return LongTsFrameAdapter(
                df, column_id, column_kind, column_value, column_sort
            )
        else:
            if column_value is not None:
                return WideTsFrameAdapter(df, column_id, column_sort, [column_value])
            else:
                return WideTsFrameAdapter(df, column_id, column_sort)

    elif _is_polars_dataframe(df):
        if column_kind is not None:
            return PolarsLongTsFrameAdapter(
                df, column_id, column_kind, column_value, column_sort
            )
        else:
            if column_value is not None:
                return PolarsWideTsFrameAdapter(
                    df, column_id, column_sort, [column_value]
                )
            else:
                return PolarsWideTsFrameAdapter(df, column_id, column_sort)

    elif isinstance(df, dict):
        if any(_is_polars_lazyframe(value) for value in df.values()):
            raise ValueError(
                "Dictionary values must be eager polars.DataFrame objects when using Polars input. "
                "Please call collect() on LazyFrame values before passing them to tsfresh."
            )

        contains_pandas_dataframes = any(
            isinstance(value, pd.DataFrame) for value in df.values()
        )
        contains_polars_dataframes = any(
            _is_polars_dataframe(value) for value in df.values()
        )

        if contains_pandas_dataframes and contains_polars_dataframes:
            raise ValueError(
                "Dictionary values must all be pandas.DataFrame or all be polars.DataFrame."
            )

        if contains_polars_dataframes:
            return PolarsTsDictAdapter(df, column_id, column_value, column_sort)

        return TsDictAdapter(df, column_id, column_value, column_sort)

    elif dd and isinstance(df, dd.DataFrame):
        return DaskTsAdapter(df, column_id, column_kind, column_value, column_sort)

    else:
        raise ValueError(
            "df must be a DataFrame or a dict of DataFrames. "
            "See https://tsfresh.readthedocs.io/en/latest/text/data_formats.html"
        )
