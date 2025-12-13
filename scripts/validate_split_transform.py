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
    Ensure the input is a pandas DataFrame.

    Parameters
    ----------
    data : array-like, dict, or pandas.DataFrame
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
    Load, validate, split, and preprocess the wine dataset.

    Parameters
    ----------
    raw_path : str
        Path to the folder containing the raw CSV data.
    raw_filename : str
        Filename of the raw CSV data.
    processed_path : str
        Path to save processed CSV datasets.
    preprocessor_path : str
        Path to save the fitted preprocessor object.
    seed : int
        Random seed for reproducibility.
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
    pickle.dump(preprocessor, open(os.path.join(preprocessor_path, "wine_preprocessor.pickle"), "wb"))

    X_train = preprocessor.fit_transform(train_df.drop(columns=["target"]))
    X_test = preprocessor.transform(test_df.drop(columns=["target"]))
    y_train = train_df["target"]
    y_test = test_df["target"]

    # export scaled features and targets
    X_train.to_csv(os.path.join(processed_path, "scaled-wine-features-train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_path, "scaled-wine-features-test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_path, "wine-target-train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_path, "wine-target-test.csv"), index=False)


@click.command()
@click.option("--raw-path", default="data/raw")
@click.option("--raw-filename", default="wine-raw.csv")
@click.option("--processed-path", default="data/processed")
@click.option("--preprocessor-path", default="results/models")
@click.option("--seed", default=123)
def main(raw_path, raw_filename, processed_path, preprocessor_path, seed):
    preprocess(raw_path, raw_filename, processed_path, preprocessor_path, seed)


if __name__ == "__main__":
    main()
