# validate_split_transform.py
# author: Harrison Li
# date: 2025-12-03

import os
import click
import numpy as np
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate import validate

def df(data):
    """
    ensure the data is pandas data frame,
    convert to pandas data frame if not
    """
    if isinstance(data, pd.DataFrame):
        return data
    else:
        return pd.DataFrame(data)

@click.command()
@click.option('--raw-path', type=str, 
              help="Path to raw data", default='~/data/raw')
@click.option('--raw-filename', type=str, 
              help="Raw data filename", default='/wine-raw.csv')
@click.option('--processed-path', type=str, 
              help="Path to export processed data", default='~/data/processed')
@click.option('--preprocessor-path', type=str, 
              help="Path to export preprocessor", default='~/results/models')
@click.option('--seed', type=int, 
              help="Random seed", default=123)
def main(raw_path, raw_filename, processed_path, preprocessor_path, seed):
    """
    This script does the following
    - loads data and validates the data to be used in exploratory data analysis
    - splits data into train/test
    - transform data for the classification model
    """
    np.random.seed(seed)
    set_config(transform_output="pandas")
    raw_path = os.path.expanduser(raw_path)
    processed_path = os.path.expanduser(processed_path)
    preprocessor_path = os.path.expanduser(preprocessor_path)
    
    # load data
    data = pd.read_csv(raw_path + raw_filename)

    # drop quality if it exists
    # the continuous target quality is not within our scope of interest in the classification model
    if 'quality' in data.columns:
        data = data.drop(columns=['quality'])

    # rename color to wine_type
    data = data.rename(columns={"color":"wine_type"})
    data["target"] = data["wine_type"].replace({
        "red": 0, 
        "white": 1
    }).astype(int)

    # validate raw data and drop duplicates if exists
    data = validate(data)

    # drop wine_type as it is encoded in target already
    data = data.drop(columns=["wine_type"])

    # split data
    train_df, test_df = train_test_split(data, test_size = 0.2)
    train_df.to_csv(processed_path + '/wine-train.csv', index=False)
    test_df.to_csv(processed_path + '/wine-test.csv', index=False)

    # preprocessor
    preprocessor = StandardScaler()

    pickle.dump(preprocessor, 
                open(os.path.join(preprocessor_path, 
                                  "wine_preprocessor.pickle"), "wb"))

    
    X_train, X_test = train_df.drop(columns=["target"]), test_df.drop(columns=["target"])
    y_train, y_test = train_df["target"], test_df["target"]

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # export splitted and transformed feature/targets
    X_train.to_csv(processed_path + '/scaled-wine-features-train.csv', index=False)
    X_test.to_csv(processed_path + '/scaled-wine-features-test.csv', index=False)
    y_train.to_csv(processed_path + '/wine-target-train.csv', index=False)
    y_test.to_csv(processed_path + '/wine-target-test.csv', index=False)

if __name__ == '__main__':
    main()