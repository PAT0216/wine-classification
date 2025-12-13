"""
Tests for the `validate` function in `src.validate` for the wine dataset.

This module uses `pytest` and `pandera` to validate that the `validate` function:

- Raises appropriate errors for invalid inputs:
  - Non-DataFrame input
  - Empty DataFrame
  - Missing columns
  - Invalid categorical labels
  - Missing values in critical columns
  - Numeric columns out of allowed ranges
  - Wrong numeric types
- Correctly handles duplicate rows by removing them
- Correctly raises errors for entirely missing observations (rows of NaNs)
- Passes when given valid data

The tests include:
1. Type checks
2. Empty DataFrame checks
3. Missing column checks
4. Invalid categorical label checks
5. Missing value checks
6. Numeric out-of-range checks (too high / too low)
7. Numeric type checks
8. Duplicate row handling
9. Entirely missing row checks
10. Parametrized tests for multiple invalid data cases
11. Validation of correctly formatted data

Examples
--------
Run all tests:

$ pytest tests/test_validation.py
"""

import pytest
import os
import numpy as np
import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.validate import validate

# -------------------------------
# Test data setup for wine dataset
# -------------------------------

valid_data = pd.DataFrame(
    {
        "fixed_acidity": [7.4, 6.0, 12.0],
        "volatile_acidity": [0.7, 0.3, 0.5],
        "citric_acid": [0.0, 0.35, 0.6],
        "residual_sugar": [1.9, 5.1, 12.5],
        "chlorides": [0.076, 0.045, 0.08],
        "free_sulfur_dioxide": [11.0, 14.0, 35.0],
        "total_sulfur_dioxide": [34.0, 132.0, 230.0],
        "density": [0.9978, 0.996, 1.005],
        "pH": [3.51, 3.2, 3.8],
        "sulphates": [0.56, 0.65, 0.7],
        "alcohol": [9.4, 10.5, 14.0],
        "wine_type": ["red", "white", "red"],
        "target": [0, 1, 0],
    }
)

# Case: wrong type passed to function
valid_data_as_np = valid_data.copy().to_numpy()


def test_valid_data_type():
    with pytest.raises(AttributeError):
        validate(valid_data_as_np)


# Case: empty data frame
empty_data = valid_data.copy().iloc[0:0]


def test_empty_data_frame():
    with pytest.raises(ValueError):
        validate(empty_data)


# -------------------------------
# Setup list of invalid data cases
# -------------------------------

invalid_data_cases = []


# -------------------------------
# Missing column tests
# -------------------------------
@pytest.mark.parametrize("col", valid_data.columns)
def test_missing_column(col):
    df_missing = valid_data.drop(columns=[col])
    with pytest.raises(ValueError, match=f"Missing columns: .*{col}.*"):
        validate(df_missing)


# Case: wrong categorical label
case_wrong_label = valid_data.copy()
case_wrong_label.loc[0, "wine_type"] = "pink"
invalid_data_cases.append((case_wrong_label, "Invalid category in 'wine_type'"))


# Case: missing value in 'wine_type' column
case_missing_wine_type = valid_data.copy()
case_missing_wine_type.loc[0, "wine_type"] = None


def test_missing_wine_type():
    with pytest.raises(
        ValueError, match="Unexpected missing values found in the dataset"
    ):
        validate(case_missing_wine_type)


# Case: numeric column out of range (too high)
numeric_cols = valid_data.select_dtypes(include=np.number).columns
for col in numeric_cols:
    case_too_high = valid_data.copy()
    case_too_high[col] = case_too_high[col] + 1000
    invalid_data_cases.append((case_too_high, f"Numeric value too high in '{col}'"))

# Case: numeric column out of range (too low)
for col in numeric_cols:
    case_too_low = valid_data.copy()
    case_too_low[col] = case_too_low[col] - 1000
    invalid_data_cases.append((case_too_low, f"Numeric value too low in '{col}'"))

# Case: wrong numeric type
for col in numeric_cols:
    case_wrong_type = valid_data.copy()
    case_wrong_type[col] = case_wrong_type[col].astype(str)
    invalid_data_cases.append(
        (case_wrong_type, f"Wrong type in numeric column '{col}'")
    )

# Case: duplicate rows (duplicates are dropped, no error should be raised)
case_duplicate = pd.concat([valid_data, valid_data.iloc[[0]]], ignore_index=True)


def test_duplicates_are_dropped():
    result = validate(case_duplicate)
    # duplicate row should be removed
    assert result.shape[0] == valid_data.shape[0]


# Case: entire row of missing values
case_missing_obs = pd.concat(
    [
        valid_data,
        pd.DataFrame([[np.nan] * valid_data.shape[1]], columns=valid_data.columns),
    ],
    ignore_index=True,
)


def test_row_all_nan():
    with pytest.raises(
        ValueError, match="Unexpected missing values found in the dataset"
    ):
        validate(case_missing_obs)


# -------------------------------
# Parametrized test for invalid data
# -------------------------------


@pytest.mark.parametrize("invalid_data, description", invalid_data_cases)
def test_invalid_wine_data(invalid_data, description):
    with pytest.raises(pa.errors.SchemaErrors):
        validate(invalid_data)


# -------------------------------
# Test valid data passes validation
# -------------------------------


def test_valid_wine_data_passes():
    result = validate(valid_data.copy())
    pd.testing.assert_frame_equal(result, valid_data)
