# tsfresh-polars

This repository is an experimental fork of [tsfresh](https://github.com/blue-yonder/tsfresh)
that keeps the familiar Python API while making eager Polars the default data format.

> **Status:** This fork is not a maintained upstream replacement for tsfresh. Check the
> source and tests before relying on it in production.

## Installation

```bash
pip install tsfresh-polars
```

Polars is installed as a runtime dependency. The package also retains pandas because several
compatibility APIs and feature calculators depend on pandas semantics.

## Basic usage

```python
import polars as pl
from tsfresh import extract_features

signals = pl.DataFrame(
    {
        "id": [1, 1, 2, 2],
        "time": [1, 2, 1, 2],
        "value": [1.0, 2.0, 3.0, 4.0],
    }
)

features = extract_features(
    signals,
    column_id="id",
    column_sort="time",
    column_value="value",
)
```

`extract_features` accepts pandas DataFrames, Polars DataFrames, and supported dictionary
formats. Its default result is a Polars DataFrame with the identifier in a regular column.
Use `output_format="pandas"` when a pandas result is required:

```python
pandas_features = extract_features(
    signals,
    column_id="id",
    column_sort="time",
    column_value="value",
    output_format="pandas",
)
```

The extraction boundary is Polars-first: Polars inputs stay in the Polars data path and
default results are Polars. Feature calculators themselves receive NumPy arrays in the
usual case, or temporary pandas Series for calculators that require pandas index semantics.

## API compatibility

| API | Polars support | Actual behavior |
| --- | --- | --- |
| `extract_features` | Native default | Polars input and Polars output by default; pandas output is opt-in. |
| `select_features` | Adapter only | Accepts a Polars feature table, converts it to pandas internally, and returns pandas. Targets remain pandas Series or NumPy arrays. |
| `extract_relevant_features` | Adapter only | Uses pandas for imputation and feature selection and returns pandas. |
| `FeatureAugmenter` | No Polars-native transformer | Keeps the original pandas/sklearn contract and requests pandas extraction output. |
| `RelevantFeatureAugmenter` | No Polars-native transformer | Uses the pandas feature-selection pipeline. |
| `impute`, `impute_dataframe_zero`, `impute_dataframe_range` | No | These utilities operate on pandas DataFrames. |
| `run_tsfresh` | No | The CLI keeps its existing pandas/CSV workflow. |

There are currently no separate Polars-native equivalents for the pandas-dependent APIs
listed above. Convert explicitly at those boundaries, or use `extract_features` directly
when a Polars result is sufficient.

## What differs from upstream tsfresh

This fork adds:

- Polars input support for wide, long, and dictionary data formats.
- Polars-first extraction with `output_format="pandas"` as the compatibility escape hatch.
- Fixes for integer overflow in arithmetic, change, energy, and entropy features.
- Fixes for spectral kurtosis, `mean_n_absolute_max` boundary handling, comprehensive
  parameter generation, categorical-ID grouping, and NumPy target alignment.

The package version and release notes are in `setup.cfg` and `CHANGES.rst`.
