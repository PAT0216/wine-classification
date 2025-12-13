"""
Module for loading, validating, splitting, and transforming the wine dataset
for classification tasks.

This script performs the following steps:
1. Loads raw wine data from a CSV file.
2. Validates the dataset using `validate` from `src.validate`.
3. Encodes wine type as a binary target column.
4. Splits the data into train and test sets.
5. Standardizes the features using `StandardScaler`.
6. Saves processed datasets and the preprocessor for future use.

Author
------
Harrison Li

Date
----
2025-12-03
"""

import os
import click
import numpy as np
import pickle
import warnings
import pandas as pd
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.validate import validate


def df(data):
    """
    Ensure the input is a pandas DataFrame.

    If the input is already a pandas DataFrame, it is returned as is.
    Otherwise, the input is converted to a DataFrame.

    Parameters
    ----------
    data : array-like, dict, or pandas.DataFrame
        Input data to be converted into a pandas DataFrame. Can be a list,
        dictionary, numpy array, or already a DataFrame.

    Returns
    -------
    pandas.DataFrame
        The input data as a pandas DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df([1, 2, 3])
       0
    0  1
    1  2
    2  3

    >>> df({'a': [1, 2], 'b': [3, 4]})
       a  b
    0  1  3
    1  2  4

    >>> df(pd.DataFrame({'x':[1,2]}))
       x
    0  1
    1  2
    """
    if isinstance(data, pd.DataFrame):
        return data
    else:
        return pd.DataFrame(data)


@click.command()
@click.option("--raw-path", type=str, help="Path to raw data", default="data/raw")
@click.option(
    "--raw-filename", type=str, help="Raw data filename", default="wine-raw.csv"
)
@click.option(
    "--processed-path",
    type=str,
    help="Path to export processed data",
    default="data/processed",
)
@click.option(
    "--preprocessor-path",
    type=str,
    help="Path to export preprocessor",
    default="results/models",
)
@click.option("--seed", type=int, help="Random seed", default=123)
def main(raw_path, raw_filename, processed_path, preprocessor_path, seed):
    """
    Load, validate, split, and preprocess the wine dataset for classification.

    The function executes the following pipeline:
    1. Load raw data from CSV.
    2. Drop the 'quality' column if present.
    3. Rename 'color' column to 'wine_type' and create a binary 'target'.
    4. Validate the dataset using `validate()`.
    5. Drop 'wine_type' after encoding.
    6. Split the dataset into train and test sets (80/20 split).
    7. Standardize the feature columns using `StandardScaler`.
    8. Save processed datasets and the preprocessor object as pickle/CSV.

    Parameters
    ----------
    raw_path : str
        Path to the folder containing the raw CSV data. Default is 'data/raw'.
    raw_filename : str
        Filename of the raw CSV data. Default is 'wine-raw.csv'.
    processed_path : str
        Path to save processed CSV datasets. Default is 'data/processed'.
    preprocessor_path : str
        Path to save the fitted preprocessor object. Default is 'results/models'.
    seed : int
        Random seed for reproducibility. Default is 123.

    Returns
    -------
    None
        The function writes processed datasets and the preprocessor to disk.

    Notes
    -----
    - The continuous 'quality' target column is dropped because this pipeline
      focuses on classification using wine type as a target.
    - Wine type is encoded as binary target: red=0, white=1.
    - StandardScaler is fitted on training features and applied to test features.
    - Outputs:
        - Processed train/test CSV files with features and target
        - StandardScaler preprocessor as a pickle file

    Examples
    --------
    Run from command line:

    $ python validate_split_transform.py \
        --raw-path data/raw \
        --raw-filename wine-raw.csv \
        --processed-path data/processed \
        --preprocessor-path results/models \
        --seed 123
    """
    np.random.seed(seed)
    set_config(transform_output="pandas")
    raw_path = os.path.expanduser(raw_path)
    processed_path = os.path.expanduser(processed_path)
    preprocessor_path = os.path.expanduser(preprocessor_path)

    # load data
    data = pd.read_csv(os.path.join(raw_path, raw_filename))

    # drop quality if it exists
    # the continuous target quality is not within our scope of interest in the classification model
    if "quality" in data.columns:
        data = data.drop(columns=["quality"])

    # rename color to wine_type
    data = data.rename(columns={"color": "wine_type"})
    data["target"] = data["wine_type"].replace({"red": 0, "white": 1}).astype(int)

    # validate raw data and drop duplicates if exists
    data = validate(data)

    # drop wine_type as it is encoded in target already
    data = data.drop(columns=["wine_type"])

    # split data
    train_df, test_df = train_test_split(data, test_size=0.2)
    train_df.to_csv(os.path.join(processed_path, "wine-train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_path, "wine-test.csv"), index=False)

    # preprocessor
    preprocessor = StandardScaler()

    pickle.dump(
        preprocessor,
        open(os.path.join(preprocessor_path, "wine_preprocessor.pickle"), "wb"),
    )

    X_train, X_test = train_df.drop(columns=["target"]), test_df.drop(
        columns=["target"]
    )
    y_train, y_test = train_df["target"], test_df["target"]

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # export splitted and transformed feature/targets
    X_train.to_csv(
        os.path.join(processed_path, "scaled-wine-features-train.csv"), index=False
    )
    X_test.to_csv(
        os.path.join(processed_path, "scaled-wine-features-test.csv"), index=False
    )
    y_train.to_csv(os.path.join(processed_path, "wine-target-train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_path, "wine-target-test.csv"), index=False)


if __name__ == "__main__":
    main()
