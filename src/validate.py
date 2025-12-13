# src/validate.py
import os
import pandera.pandas as pa
from pandera.errors import SchemaErrors
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"


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