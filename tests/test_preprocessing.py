"""
Tests for the wine data preprocessing pipeline.

Attribution: the help of ChatGPT was used to cover various test cases and correct for errors.

This module contains pytest-based tests to verify that the preprocessing
pipeline defined in ``scripts.validate_split_transform`` behaves correctly.
Specifically, it checks that:

- Raw CSV data can be loaded and processed without errors.
- All expected output files (processed CSVs and preprocessor pickle)
  are created in the correct locations.
- Exported CSV files contain consistent and non-empty data.
- The saved preprocessor is of the expected type.
- The utility function ``df`` correctly converts inputs to pandas DataFrames.

These tests rely on pytest fixtures and temporary directories to ensure
that no real data or filesystem paths are modified during execution.
"""
import pytest
import os
import pandas as pd
import pickle
import warnings
from scripts.validate_split_transform import preprocess, df
from sklearn.preprocessing import StandardScaler
warnings.simplefilter(action="ignore", category=FutureWarning)

@pytest.fixture
def sample_raw_csv(tmp_path):
    """
    Create a temporary raw wine CSV file for testing.

    The fixture generates a small, valid wine dataset containing all required
    columns and writes it to a temporary directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    tuple of str
        A tuple containing:
        - Path to the raw data directory.
        - Filename of the raw CSV file.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df_raw = pd.DataFrame(
        {
            "color": ["red", "white", "red"],
            "fixed_acidity": [7.4, 6.0, 12.0],
            "volatile_acidity": [0.7, 0.3, 0.5],
            "citric_acid": [0.0, 0.35, 0.6],
            "residual_sugar": [1.9, 5.1, 12.5],
            "chlorides": [0.076, 0.045, 0.08],
            "free_sulfur_dioxide": [11., 14., 35.],
            "total_sulfur_dioxide": [34., 132., 230.],
            "density": [0.9978, 0.996, 1.005],
            "pH": [3.51, 3.2, 3.8],
            "sulphates": [0.56, 0.65, 0.7],
            "alcohol": [9.4, 10.5, 14.0],
            "quality": [5., 6., 7.],
        }
    )
    raw_file = raw_dir / "wine-raw.csv"
    df_raw.to_csv(raw_file, index=False)
    return str(raw_dir), raw_file.name


@pytest.fixture
def processed_dir(tmp_path):
    """
    Create a temporary directory for processed CSV outputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    str
        Path to the processed data directory.
    """
    d = tmp_path / "processed"
    d.mkdir()
    return str(d)


@pytest.fixture
def preproc_dir(tmp_path):
    """
    Create a temporary directory for the preprocessor.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    str
        Path to the preprocessor directory.
    """
    d = tmp_path / "preprocessor"
    d.mkdir()
    return str(d)

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_preprocessing_exports(sample_raw_csv, processed_dir, preproc_dir):
    """
    Test that the preprocessing exports all expected files.

    This test runs the preprocessing and verifies that:
    - All expected CSV files are created.
    - The preprocessor pickle file exists.
    - The saved preprocessor is an instance of ``StandardScaler``.
    """
    raw_dir, raw_filename = sample_raw_csv
    preprocess(raw_dir, raw_filename, processed_dir, preproc_dir, seed=123)

    expected_files = [
        "wine-train.csv",
        "wine-test.csv",
        "scaled-wine-features-train.csv",
        "scaled-wine-features-test.csv",
        "wine-target-train.csv",
        "wine-target-test.csv",
        "wine_preprocessor.pickle",
    ]
    for fname in expected_files:
        path = os.path.join(
            processed_dir if fname.endswith(".csv") else preproc_dir, fname
        )
        assert os.path.exists(path), f"{fname} not found"

    # Check preprocessor type
    preproc_file = os.path.join(preproc_dir, "wine_preprocessor.pickle")
    with open(preproc_file, "rb") as f:
        preprocessor = pickle.load(f)
    assert isinstance(preprocessor, StandardScaler)

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_csv_contents(sample_raw_csv, processed_dir, preproc_dir):
    """
    Test that exported CSV files contain valid and consistent data.

    This test ensures that:
    - Feature and target CSV files are not empty.
    - The number of feature rows matches the number of target rows
      for both training and test splits.
    """
    raw_dir, raw_filename = sample_raw_csv
    preprocess(raw_dir, raw_filename, processed_dir, preproc_dir, seed=123)

    X_train = pd.read_csv(os.path.join(processed_dir, "scaled-wine-features-train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "scaled-wine-features-test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "wine-target-train.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "wine-target-test.csv"))

    # Basic sanity checks
    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_df_transform():
    """
    Test the ``df`` utility function.

    This test verifies that:
    - List inputs are converted to DataFrames.
    - Dictionary inputs are converted to DataFrames with correct columns.
    - DataFrame inputs are returned unchanged.
    """
    # List input
    df1 = df([1, 2, 3])
    assert isinstance(df1, pd.DataFrame)
    # Dict input
    df2 = df({"a": [1, 2], "b": [3, 4]})
    assert isinstance(df2, pd.DataFrame)
    assert list(df2.columns) == ["a", "b"]
    # DataFrame input
    df3 = pd.DataFrame({"x": [5, 6]})
    df4 = df(df3)
    pd.testing.assert_frame_equal(df3, df4)
