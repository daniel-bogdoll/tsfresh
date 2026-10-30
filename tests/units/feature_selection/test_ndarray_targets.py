import numpy as np
import pandas as pd

from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.transformers.feature_selector import FeatureSelector


def test_relevance_table_aligns_ndarray_target_to_x_index():
    X = pd.DataFrame({"signal": [0, 1, 1], "noise": [0, 0, 1]}, index=[10, 5, 3])
    y = np.array([0, 1, 1])
    ndarray_result = calculate_relevance_table(X, y, n_jobs=0)
    series_result = calculate_relevance_table(
        X, pd.Series(y, index=X.index), n_jobs=0
    )
    pd.testing.assert_frame_equal(ndarray_result, series_result)


def test_feature_selector_aligns_ndarray_target_to_x_index():
    y = np.tile([0, 1], 10)
    X = pd.DataFrame(
        {"signal": y, "noise": np.tile([0, 0, 1, 1], 5)},
        index=np.arange(20) + 100,
    )
    selector = FeatureSelector(n_jobs=0).fit(X, y)
    assert selector.relevant_features == ["signal"]
