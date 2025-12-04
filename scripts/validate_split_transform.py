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

import pandera.pandas as pa
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
from pandera.errors import SchemaErrors

from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def validate(data):
    """ Validates raw data
    - check if data is empty
    - check for expected columns
    pandera:
    - check for missing values
    - check for correct data type
    - check for / drop duplicated observations
    - check for outliers or anonymous values
    - check for correct category levels
    """
    # check if data is empty
    if len(data) == 0:
        raise ValueError("The dataset is empty.")

    # check for expected columns
    expected_columns = {
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol", "wine_type", "target"
    }
    if not expected_columns.issubset(data.columns):
        missing = expected_columns - set(data.columns)
        raise ValueError(f"Missing columns: {missing}")

    # The notebook mentions raw data has no NAs, but good to verify
    if data.isnull().any().any():
        raise ValueError("Unexpected missing values found in the dataset.")

    # pandera check
    # | error: True
    # the range of checks is determined by referring to the paper related to the dataset
    schema = pa.DataFrameSchema(
        {   
            # data type check
            "fixed_acidity": pa.Column(float, pa.Check.between(0, 20), nullable=True), 
            "volatile_acidity": pa.Column(float, pa.Check.between(0, 2), nullable=True), 
            "citric_acid": pa.Column(float, pa.Check.between(0, 2), nullable=True), 
            "residual_sugar": pa.Column(float, pa.Check.between(0, 100), nullable=True),
            "chlorides": pa.Column(float, pa.Check.between(0, 1), nullable=True), 
            "free_sulfur_dioxide": pa.Column(float, pa.Check.between(0, 500), nullable=True), 
            "total_sulfur_dioxide": pa.Column(float, pa.Check.between(0, 600), nullable=True), 
            "density": pa.Column(float, pa.Check.between(0.8, 1.2), nullable=True),
            "pH": pa.Column(float, pa.Check.between(0, 6), nullable=True), 
            "sulphates": pa.Column(float, pa.Check.between(0, 3), nullable=True), 
            "alcohol": pa.Column(float, pa.Check.between(5, 20), nullable=True), 
            
            # targets should not be nullable
            "wine_type": pa.Column(str, pa.Check.isin(['white', 'red']), nullable=False), 
            "target": pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
        },
        checks=[
            # duplicate observation check
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ]
    )
    
    try:
        schema.validate(data, lazy=True)
    except SchemaErrors as se:
        if "Duplicate rows found." in str(se):
            # print("Duplicate rows detected — drop duplicates and re-validate.")
            data = data.drop_duplicates()
            schema.validate(data, lazy=True)  # re-validate after dropping duplicates
        else:
            # raise other SchemaErrors if it is other issue
            raise se
    
    # # uncomment the print statement below to display test message
    # print("pass all validation")
    
    return data

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