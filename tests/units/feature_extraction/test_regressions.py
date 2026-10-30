import math

import numpy as np
import pandas as pd
import pytest

from tsfresh import extract_features
from tsfresh.feature_extraction.feature_calculators import (
    abs_energy,
    approximate_entropy,
    fft_aggregated,
    mean_abs_change,
    mean_change,
    mean_n_absolute_max,
    root_mean_square,
    sample_entropy,
)
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters


def test_integer_arithmetic_features_do_not_overflow():
    values = np.array([10, 9, 8, 7], dtype=np.uint8)
    assert mean_change(values) == -1
    assert mean_abs_change(values) == 1
    assert abs_energy(np.array([20000, 30000, -20000, -30000], dtype=np.int16)) == 2.6e9
    assert root_mean_square(np.array([20000, 30000, -20000, -30000], dtype=np.int16)) == math.sqrt(
        650000000
    )


def test_entropy_features_handle_integer_and_boolean_inputs():
    values = np.array([True, False, True, True, False, False] * 3)
    np.testing.assert_allclose(
        sample_entropy(values), sample_entropy(values.astype(float))
    )
    np.testing.assert_allclose(
        approximate_entropy(values, 2, 0.2),
        approximate_entropy(values.astype(float), 2, 0.2),
    )


def test_fft_kurtosis_and_mean_n_absolute_max_boundaries():
    result = dict(
        fft_aggregated(
            np.arange(10),
            [{"aggtype": "kurtosis"}],
        )
    )
    assert result['aggtype_"kurtosis"'] == pytest.approx(3.362, abs=1e-3)
    assert mean_n_absolute_max([-2, 3, -4], 3) == 3


def test_comprehensive_settings_keep_all_mean_n_absolute_max_parameters():
    params = ComprehensiveFCParameters()["mean_n_absolute_max"]
    assert [param["number_of_maxima"] for param in params] == [3, 5, 7]


def test_categorical_ids_do_not_create_unused_rows():
    frame = pd.DataFrame(
        {
            "id": pd.Categorical([1, 3, 1, 3], categories=[1, 2, 3]),
            "time": [2, 2, 1, 1],
            "value": [4, 8, 2, 6],
        }
    )
    result = extract_features(
        frame,
        column_id="id",
        column_sort="time",
        default_fc_parameters={"length": None},
        n_jobs=0,
        disable_progressbar=True,
    )
    assert result.index.tolist() == [1, 3]
