# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
import tempfile
from unittest import skipUnless
from unittest.mock import patch

import numpy as np
import pandas as pd
from mock import Mock

from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters,
    PickableSettings,
)
from tsfresh.utilities.distribution import IterableDistributorBaseClass, MapDistributor

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


class ExtractionTestCase(DataTestCase):
    """The unit tests in this module make sure if the time series features are created properly"""

    def setUp(self):
        self.n_jobs = 1
        self.directory = tempfile.gettempdir()

    def test_extract_features(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        )
        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(
            np.all(extracted_features.a__sum_values == np.array([691, 1017]))
        )
        self.assertTrue(
            np.all(extracted_features.a__abs_energy == np.array([32211, 63167]))
        )
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))

        df_sts = self.create_one_valued_time_series()
        extracted_features_sts = extract_features(
            df_sts,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        )

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(
            np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0]))
        )
        self.assertTrue(
            np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0]))
        )
        self.assertTrue(
            np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1]))
        )

    def test_extract_features_uses_only_kind_to_fc_settings(self):
        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
            kind_to_fc_parameters={"a": {"maximum": None, "minimum": None}},
        )
        assert len(extracted_features) == 2

    def test_extract_features_for_one_time_series(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        settings = ComprehensiveFCParameters()

        extracted_features = extract_features(
            df,
            default_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))

        df_sts = self.create_one_valued_time_series()
        extracted_features_sts = extract_features(
            df_sts,
            default_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
        )

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(
            np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0]))
        )
        self.assertTrue(
            np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0]))
        )
        self.assertTrue(
            np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1]))
        )

    def test_extract_features_for_index_based_functions(self):
        df = self.create_test_data_sample_with_time_index()

        settings = {
            "linear_trend_timewise": [{"attr": "slope"}],
            "linear_trend": [{"attr": "slope"}],
        }

        extracted_features = extract_features(
            df,
            default_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)

        slope_a = extracted_features['a__linear_trend_timewise__attr_"slope"'].values
        slope_b = extracted_features['b__linear_trend_timewise__attr_"slope"'].values

        self.assertAlmostEqual(slope_a[0], -0.001347117)
        self.assertAlmostEqual(slope_a[1], 0.052036340)
        self.assertAlmostEqual(slope_b[0], 0.021898496)
        self.assertAlmostEqual(slope_b[1], -0.012312)

        # Test that the index of the returned df is the ID and not the timestamp
        self.assertTrue(extracted_features.index.dtype != df.index.dtype)
        self.assertTrue(extracted_features.index.dtype == df["id"].dtype)
        self.assertEqual(
            sorted(extracted_features.index.unique().tolist()),
            sorted(df["id"].unique().tolist()),
        )

    def test_extract_features_custom_function(self):
        df = self.create_test_data_sample()

        def custom_function(x, p):
            return len(x) + p

        settings = PickableSettings(
            {
                "mean": None,
                custom_function: [{"p": 1}, {"p": -1}],
            }
        )

        extracted_features = extract_features(
            df,
            default_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)

        mean_a = extracted_features["a__mean"].values
        custom_function_a_1 = extracted_features["a__custom_function__p_1"].values
        custom_function_a_m1 = extracted_features["a__custom_function__p_-1"].values

        self.assertAlmostEqual(mean_a[0], 34.55)
        self.assertAlmostEqual(mean_a[1], 50.85)
        self.assertAlmostEqual(custom_function_a_1[0], 21)
        self.assertAlmostEqual(custom_function_a_1[1], 21)
        self.assertAlmostEqual(custom_function_a_m1[0], 19)
        self.assertAlmostEqual(custom_function_a_m1[1], 19)

    def test_extract_features_after_randomisation(self):
        df = self.create_test_data_sample()
        df_random = df.copy().sample(frac=1)

        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        ).sort_index()
        extracted_features_from_random = extract_features(
            df_random,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        ).sort_index()

        self.assertCountEqual(
            extracted_features.columns, extracted_features_from_random.columns
        )

        for col in extracted_features:
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    extracted_features[col], extracted_features_from_random[col]
                )
            )

    def test_profiling_file_written_out(self):

        df = pd.DataFrame(
            data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)}
        )
        profiling_filename = os.path.join(self.directory, "test_profiling.txt")
        X = extract_features(
            df,
            column_id="id",
            column_value="val",
            n_jobs=self.n_jobs,
            profile=True,
            profiling_filename=profiling_filename,
        )

        self.assertTrue(os.path.isfile(profiling_filename))
        os.remove(profiling_filename)

    def test_profiling_cumulative_file_written_out(self):

        PROFILING_FILENAME = os.path.join(
            self.directory, "test_profiling_cumulative.txt"
        )
        PROFILING_SORTING = "cumulative"

        df = pd.DataFrame(
            data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)}
        )
        extract_features(
            df,
            column_id="id",
            column_value="val",
            n_jobs=self.n_jobs,
            profile=True,
            profiling_filename=PROFILING_FILENAME,
            profiling_sorting=PROFILING_SORTING,
        )

        self.assertTrue(os.path.isfile(PROFILING_FILENAME))
        os.remove(PROFILING_FILENAME)

    def test_extract_features_without_settings(self):
        df = pd.DataFrame(
            data={
                "id": np.repeat([1, 2], 10),
                "value1": np.random.normal(0, 1, 20),
                "value2": np.random.normal(0, 1, 20),
            }
        )
        X = extract_features(df, column_id="id", n_jobs=self.n_jobs)
        self.assertIn("value1__maximum", list(X.columns))
        self.assertIn("value2__maximum", list(X.columns))

    def test_extract_features_with_and_without_parallelization(self):
        df = self.create_test_data_sample()

        features_parallel = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=2,
        )

        features_serial = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=0,
        )

        self.assertCountEqual(features_parallel.columns, features_serial.columns)

        for col in features_parallel.columns:
            np.testing.assert_array_almost_equal(
                features_parallel[col], features_serial[col]
            )

    def test_extract_index_preservation(self):
        df = self.create_test_data_nearly_numerical_indices()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertEqual(set(df["id"]), set(extracted_features.index))

    def test_extract_features_alphabetically_sorted(self):
        df = self.create_test_data_sample()

        features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
        )

        for col_name in features.columns:
            # split out the configuration of the features calculator
            col_name_chunks = col_name.split("__")
            # the name is always at the beginning, so remove it. Also remove the kind of the column
            col_name_chunks = col_name_chunks[2:]

            self.assertEqual(col_name_chunks, list(sorted(col_name_chunks)))


class ParallelExtractionTestCase(DataTestCase):
    def setUp(self):
        self.n_jobs = 2

        # only calculate some features to reduce load on travis ci
        self.name_to_param = {
            "maximum": None,
            "sum_values": None,
            "abs_energy": None,
            "minimum": None,
            "mean": None,
            "median": None,
        }

    def test_extract_features(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(
            np.all(extracted_features.a__sum_values == np.array([691, 1017]))
        )
        self.assertTrue(
            np.all(extracted_features.a__abs_energy == np.array([32211, 63167]))
        )
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))


class DistributorUsageTestCase(DataTestCase):
    def setUp(self):
        # only calculate some features to reduce load on travis ci
        self.name_to_param = {"maximum": None}

    def test_distributor_map_reduce_is_called(self):
        df = self.create_test_data_sample()

        mock = Mock(spec=IterableDistributorBaseClass)
        mock.close.return_value = None
        mock.map_reduce.return_value = []

        X = extract_features(
            timeseries_container=df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            default_fc_parameters=self.name_to_param,
            distributor=mock,
        )

        self.assertTrue(mock.map_reduce.called)

    def test_distributor_close_is_called(self):
        df = self.create_test_data_sample()

        mock = MapDistributor()
        mock.close = Mock()
        mock.close.return_value = None

        X = extract_features(
            timeseries_container=df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            default_fc_parameters=self.name_to_param,
            distributor=mock,
        )

        self.assertTrue(mock.close.called)


@skipUnless(pl is not None, "polars is not installed")
class PolarsExtractionParityTestCase(DataTestCase):
    def setUp(self):
        self.fc_parameters = {"sum_values": None, "mean": None, "median": None}

    def _assert_pandas_polars_equal(self, pandas_df, polars_df, **kwargs):
        pandas_features = (
            extract_features(
                pandas_df, default_fc_parameters=self.fc_parameters, **kwargs
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        polars_features = (
            extract_features(
                polars_df,
                default_fc_parameters=self.fc_parameters,
                output_format="pandas",
                **kwargs
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        self.assertIsInstance(polars_features, pd.DataFrame)
        self.assertEqual(list(pandas_features.index), list(polars_features.index))
        self.assertEqual(pandas_features.index.name, polars_features.index.name)
        self.assertEqual(list(pandas_features.columns), list(polars_features.columns))
        np.testing.assert_allclose(
            pandas_features.to_numpy(),
            polars_features.to_numpy(),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

    def test_wide_format_parity(self):
        pandas_df = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "time": [2, 1, 3, 3, 1, 2],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        polars_df = pl.from_pandas(pandas_df)
        self._assert_pandas_polars_equal(
            polars_df=polars_df, pandas_df=pandas_df, column_id="id", column_sort="time"
        )

    def test_long_format_parity(self):
        pandas_df = pd.DataFrame(
            {
                "id": ["a", "a", "a", "a", "b", "b"],
                "time": [2, 1, 2, 1, 1, 2],
                "kind": ["x", "x", "y", "y", "x", "x"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        polars_df = pl.from_pandas(pandas_df)
        self._assert_pandas_polars_equal(
            polars_df=polars_df,
            pandas_df=pandas_df,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
        )

    def test_datetime_sort_parity(self):
        pandas_df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": pd.to_datetime(
                    [
                        "2024-01-02T00:00:00",
                        "2024-01-01T00:00:00",
                        "2024-01-02T00:00:00",
                        "2024-01-01T00:00:00",
                    ]
                ),
                "value": [3.0, 1.0, 4.0, 2.0],
            }
        )
        polars_df = pl.from_pandas(pandas_df)
        self._assert_pandas_polars_equal(
            polars_df=polars_df,
            pandas_df=pandas_df,
            column_id="id",
            column_sort="time",
            column_value="value",
        )

    def test_dict_parity(self):
        pandas_dict = {
            "a": pd.DataFrame(
                {"id": [1, 1, 2], "time": [2, 1, 1], "value": [1.0, 2.0, 3.0]}
            ),
            "b": pd.DataFrame(
                {"id": [1, 2, 2], "time": [1, 2, 1], "value": [4.0, 5.0, 6.0]}
            ),
        }
        polars_dict = {kind: pl.from_pandas(df) for kind, df in pandas_dict.items()}

        pandas_features = (
            extract_features(
                pandas_dict,
                default_fc_parameters=self.fc_parameters,
                column_id="id",
                column_sort="time",
                column_value="value",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        polars_features = (
            extract_features(
                polars_dict,
                default_fc_parameters=self.fc_parameters,
                column_id="id",
                column_sort="time",
                column_value="value",
                output_format="pandas",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        self.assertEqual(list(pandas_features.index), list(polars_features.index))
        self.assertEqual(list(pandas_features.columns), list(polars_features.columns))
        np.testing.assert_allclose(
            pandas_features.to_numpy(),
            polars_features.to_numpy(),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

    def test_no_full_frame_to_pandas_conversion(self):
        polars_df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [2, 1, 2, 1],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with patch.object(
            pl.DataFrame,
            "to_pandas",
            side_effect=AssertionError("full-frame conversion not allowed"),
        ):
            features = extract_features(
                polars_df,
                default_fc_parameters=self.fc_parameters,
                column_id="id",
                column_sort="time",
                column_value="value",
            )

        self.assertIsInstance(features, pl.DataFrame)

    def test_default_output_matches_input_format(self):
        pandas_df = pd.DataFrame(
            {"id": [1, 1, 2, 2], "time": [2, 1, 2, 1], "value": [1.0, 2.0, 3.0, 4.0]}
        )
        polars_df = pl.from_pandas(pandas_df)

        pandas_features = extract_features(
            pandas_df,
            default_fc_parameters=self.fc_parameters,
            column_id="id",
            column_sort="time",
            column_value="value",
        )
        polars_features = extract_features(
            polars_df,
            default_fc_parameters=self.fc_parameters,
            column_id="id",
            column_sort="time",
            column_value="value",
        )

        self.assertIsInstance(pandas_features, pd.DataFrame)
        self.assertIsInstance(polars_features, pl.DataFrame)
