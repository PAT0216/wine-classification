# Attribution: the help of ChatGPT was used to cover various test cases and correct for errors.

import pandas as pd
import os
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.correlation_plot import correlation_plot

# error cases
def test_wine_data_not_dataframe_raises_type_error(tmp_path):
    df = [1, 2, 3]  # not a DataFrame
    with pytest.raises(TypeError, match="The wine_data must be a pandas DataFrame"):
        correlation_plot(df, ["a"], save_to=str(tmp_path / "plot.png"))

def test_empty_wine_data_raises_value_error(tmp_path):
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="The wine_data DataFrame cannot be empty"):
        correlation_plot(df, ["a"], save_to=str(tmp_path / "plot.png"))

def test_correlation_cols_not_list_raises_type_error(tmp_path):
    df = pd.DataFrame({"a": [1,2,3]})
    with pytest.raises(TypeError, match="The correlation_cols must be a list of column names"):
        correlation_plot(df, "a", save_to=str(tmp_path / "plot.png"))

def test_correlation_cols_empty_list_raises_value_error(tmp_path):
    df = pd.DataFrame({"a": [1,2,3]})
    with pytest.raises(ValueError, match="The correlation_cols cannot be an empty list"):
        correlation_plot(df, [], save_to=str(tmp_path / "plot.png"))

def test_correlation_cols_non_string_entry_raises_type_error(tmp_path):
    df = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
    with pytest.raises(TypeError, match="All entries in correlation_cols must be strings"):
        correlation_plot(df, ["a", 5], save_to=str(tmp_path / "plot.png"))

def test_missing_columns_raise_value_error(tmp_path):
    df = pd.DataFrame({"a": [1,2,3]})
    with pytest.raises(ValueError, match="The following columns were not found"):
        correlation_plot(df, ["a","b"], save_to=str(tmp_path / "plot.png"))

def test_save_to_not_string_raises_type_error(tmp_path):
    df = pd.DataFrame({"a": [1,2,3]})
    with pytest.raises(TypeError, match="save_to must be a string file path"):
        correlation_plot(df, ["a"], save_to=123)

# edge cases
def test_directory_created_if_missing(tmp_path):
    df = pd.DataFrame({"a": [1,2,3], "b": [2,4,6]})
    output_file = tmp_path / "nested" / "dir" / "plot.png"

    # Function should create directories automatically
    correlation_plot(df, ["a","b"], save_to=str(output_file))

    # Directory should exist
    assert output_file.parent.exists()
    # File should exist
    assert output_file.exists()

def test_single_column_dataframe(tmp_path):
    """Test that a single-column DataFrame still returns a valid correlation DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    output_file = tmp_path / "plot.png"
    corr_df = correlation_plot(df, ["a"], save_to=str(output_file))

    assert list(corr_df.columns) == ["feature_x", "feature_y", "correlation"]
    assert len(corr_df) == 1  # Only one entry for correlation of the single column
    assert output_file.exists()

# expected
def test_everything_works(tmp_path):
    """Test everything works when provided correct information."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [2, 4, 6],
        "c": [3, 6, 9]
    })
    output_file = tmp_path / "plot.png"
    corr_df = correlation_plot(df, ["a", "b", "c"], save_to=str(output_file))

    # Check DataFrame structure
    assert list(corr_df.columns) == ["feature_x", "feature_y", "correlation"]
    # Check correlation values
    assert corr_df["correlation"].round(6).eq(1.0).all()
    # Check file exists
    assert output_file.exists()
