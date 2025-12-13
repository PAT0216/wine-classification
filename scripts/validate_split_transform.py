import os
import numpy as np
import pickle
import warnings
import pandas as pd
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import click

warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.validate import validate


def df(data):
    """
    Convert input data to a pandas DataFrame if necessary.

    If the input is already a pandas DataFrame, it is returned unchanged.
    Otherwise, the input is passed to the pandas DataFrame constructor.

    Parameters
    ----------
    data : array, dict, or pandas.DataFrame
        Input data to be converted into a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data
    else:
        return pd.DataFrame(data)


def preprocess(raw_path, raw_filename, processed_path, preprocessor_path, seed=123):
    """
    Load, validate, split, and preprocess the wine dataset for classification.

    This function performs the full preprocessing pipeline:
    1. Loads raw wine data from a CSV file.
    2. Drops the `quality` column if present.
    3. Renames the `color` column to `wine_type` and encodes it as a binary target.
    4. Validates the dataset using `src.validate.validate`.
    5. Splits the data into training and test sets.
    6. Fits a `StandardScaler` on training features and applies it to test features.
    7. Saves processed datasets and the preprocessor to disk.

    Parameters
    ----------
    raw_path : str
        Path to the directory containing the raw CSV file.
    raw_filename : str
        Name of the raw CSV file.
    processed_path : str
        Directory where processed CSV files will be written.
    preprocessor_path : str
        Directory where the fitted preprocessor pickle will be saved.
    seed : int, default=123
        Random seed used for reproducible train/test splitting.

    Returns
    -------
    None
        This function writes processed data and the preprocessor to disk.

    Raises
    ------
    FileNotFoundError
        If the raw data file does not exist.
    ValueError
        If validation fails or the dataset is empty.
    pandera.errors.SchemaErrors
        If the dataset does not conform to the validation schema.

    Notes
    -----
    - Wine type is encoded as:
        - red   = 0
        - white = 1
    - Duplicate rows are removed during validation.
    - Output files written to `processed_path`:
        - wine-train.csv
        - wine-test.csv
        - scaled-wine-features-train.csv
        - scaled-wine-features-test.csv
        - wine-target-train.csv
        - wine-target-test.csv
    - Output file written to `preprocessor_path`:
        - wine_preprocessor.pickle

    Examples
    --------
    >>> preprocess(
    ...     raw_path="data/raw",
    ...     raw_filename="wine-raw.csv",
    ...     processed_path="data/processed",
    ...     preprocessor_path="results/models",
    ...     seed=123,
    ... )
    """
    np.random.seed(seed)
    set_config(transform_output="pandas")
    raw_path = os.path.expanduser(raw_path)
    processed_path = os.path.expanduser(processed_path)
    preprocessor_path = os.path.expanduser(preprocessor_path)

    # load data
    data = pd.read_csv(os.path.join(raw_path, raw_filename))

    # drop quality if it exists
    if "quality" in data.columns:
        data = data.drop(columns=["quality"])

    # rename color to wine_type and create target
    data = data.rename(columns={"color": "wine_type"})
    data["target"] = data["wine_type"].replace({"red": 0, "white": 1}).astype(int)

    # validate raw data
    data = validate(data)

    # drop wine_type as it is encoded in target
    data = data.drop(columns=["wine_type"])

    # split data
    train_df, test_df = train_test_split(data, test_size=0.2)
    os.makedirs(processed_path, exist_ok=True)
    train_df.to_csv(os.path.join(processed_path, "wine-train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_path, "wine-test.csv"), index=False)

    # preprocessor
    preprocessor = StandardScaler()
    os.makedirs(preprocessor_path, exist_ok=True)
    pickle.dump(
        preprocessor,
        open(os.path.join(preprocessor_path, "wine_preprocessor.pickle"), "wb"),
    )

    X_train = preprocessor.fit_transform(train_df.drop(columns=["target"]))
    X_test = preprocessor.transform(test_df.drop(columns=["target"]))
    y_train = train_df["target"]
    y_test = test_df["target"]

    # export scaled features and targets
    X_train.to_csv(
        os.path.join(processed_path, "scaled-wine-features-train.csv"), index=False
    )
    X_test.to_csv(
        os.path.join(processed_path, "scaled-wine-features-test.csv"), index=False
    )
    y_train.to_csv(os.path.join(processed_path, "wine-target-train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_path, "wine-target-test.csv"), index=False)


@click.command()
@click.option("--raw-path", default="data/raw")
@click.option("--raw-filename", default="wine-raw.csv")
@click.option("--processed-path", default="data/processed")
@click.option("--preprocessor-path", default="results/models")
@click.option("--seed", default=123)
def main(raw_path, raw_filename, processed_path, preprocessor_path, seed):
    """
    Command-line entry point for the wine preprocessing pipeline.

    This function wraps the `preprocess` function for CLI usage via `click`.

    Parameters
    ----------
    raw_path : str
        Path to the directory containing raw data.
    raw_filename : str
        Name of the raw CSV file.
    processed_path : str
        Output directory for processed datasets.
    preprocessor_path : str
        Output directory for the saved preprocessor.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None

    Example
    -------
    Run the preprocessing pipeline from the command line:

    >>> python validate_split_transform.py \
    ...     --raw-path data/raw \
    ...     --raw-filename wine-raw.csv \
    ...     --processed-path data/processed \
    ...     --preprocessor-path results/models \
    ...     --seed 123
    """
    preprocess(raw_path, raw_filename, processed_path, preprocessor_path, seed)


if __name__ == "__main__":
    main()
