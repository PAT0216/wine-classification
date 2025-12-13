"""
Download the Wine Quality dataset from the UCI Machine Learning Repository.

This script fetches the Wine Quality dataset using the `ucimlrepo` package
and saves the raw data as a CSV file to a specified directory.

The dataset is written as `wine-raw.csv` and is intended to be used as the
raw input for downstream preprocessing and modeling pipelines.

Author
------
Harrison Li

Date
----
2025-12-03
"""
import os
import click
import pandas as pd
from ucimlrepo import fetch_ucirepo 

@click.command()
@click.option('--path', type=str, 
              help="Path to directory where raw data will be written to", default='data/raw')
@click.option('--raw-data-filename', type=str, 
              help="Filename for the raw dataset CSV", default='wine-raw.csv')
def main(path, raw_data_filename):
    """
    Download and save the raw Wine Quality dataset.

    The Wine Quality dataset is retrieved from the UCI Machine Learning
    Repository and saved as a CSV file in the specified directory.

    Parameters
    ----------
    path : str
        Directory where the raw dataset will be written.
    raw_data_filename : str
        Name of the CSV file used to store the raw dataset.

    Notes
    -----
    This function does not perform any preprocessing or validation. It
    simply downloads and stores the dataset in its original form.

    Examples
    --------
    >>> python scripts/download_data.py \ 
    ...     --path data/raw \ 
    ...     --raw-data-filename wine-raw.csv
    """
    os.makedirs(path, exist_ok=True)
    wine_raw = fetch_ucirepo(id=186)
    wine_raw.data.original.to_csv(os.path.join(path, raw_data_filename), index=False)

if __name__ == '__main__':
    main()