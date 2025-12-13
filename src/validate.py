# src/validate.py
import os
import pandera.pandas as pa
from pandera.errors import SchemaErrors

os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"


def validate(data):
    """
    Validate a raw wine dataset for integrity, consistency, and expected ranges.

    The function performs the following checks:
    1. Ensures the dataset is not empty.
    2. Checks that all expected columns are present.
    3. Checks for unexpected missing values.
    4. Uses a `pandera` DataFrameSchema to validate:
       - Data types for each column
       - Value ranges for numerical columns
       - Allowed categories for categorical columns
       - Duplicate rows
       - Entirely empty rows

    If duplicate rows are found, they are dropped and the validation is re-run.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw wine dataset containing columns:
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol", "wine_type", "target".

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame, with duplicate rows removed if any were present.

    Raises
    ------
    ValueError
        If the dataset is empty, missing expected columns, or contains unexpected missing values.
    pandera.errors.SchemaErrors
        If any schema validation checks fail (other than duplicates, which are handled internally).

    Notes
    -----
    - Numerical columns are validated to lie within ranges informed by the dataset documentation:
        - fixed_acidity: 0-20
        - volatile_acidity: 0-2
        - citric_acid: 0-2
        - residual_sugar: 0-100
        - chlorides: 0-1
        - free_sulfur_dioxide: 0-500
        - total_sulfur_dioxide: 0-600
        - density: 0.8-1.2
        - pH: 0-6
        - sulphates: 0-3
        - alcohol: 5-20
    - Categorical columns are validated for allowed categories:
        - wine_type: {'white', 'red'}
        - target: {0, 1}
    - Duplicate rows are automatically removed and validation is re-applied.

    Examples
    --------
    >>> import pandas as pd
    >>> from validate import validate
    >>> df = pd.read_csv("wine_data.csv")
    >>> validated_df = validate(df)
    >>> validated_df.shape
    (6497, 13)
    """
    # check if data is empty
    if len(data) == 0:
        raise ValueError("The dataset is empty.")

    # check for expected columns
    expected_columns = {
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "wine_type",
        "target",
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
            "free_sulfur_dioxide": pa.Column(
                float, pa.Check.between(0, 500), nullable=True
            ),
            "total_sulfur_dioxide": pa.Column(
                float, pa.Check.between(0, 600), nullable=True
            ),
            "density": pa.Column(float, pa.Check.between(0.8, 1.2), nullable=True),
            "pH": pa.Column(float, pa.Check.between(0, 6), nullable=True),
            "sulphates": pa.Column(float, pa.Check.between(0, 3), nullable=True),
            "alcohol": pa.Column(float, pa.Check.between(5, 20), nullable=True),
            # targets should not be nullable
            "wine_type": pa.Column(
                str, pa.Check.isin(["white", "red"]), nullable=False
            ),
            "target": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        },
        checks=[
            # duplicate observation check
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(
                lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."
            ),
        ],
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
