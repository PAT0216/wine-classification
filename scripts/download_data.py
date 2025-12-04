# download_data.py
# author: Harrison Li
# date: 2025-12-03

import click
import pandas as pd
from ucimlrepo import fetch_ucirepo 

@click.command()
@click.option('--pwd', type=str, 
              help="Path to directory where raw data will be written to", default='~/data/raw')
def main(pwd):
    """downloads data to data/raw"""
    export_path = pwd + "/wine-raw.csv"
    print(export_path)
    wine_raw = fetch_ucirepo(id=186).data.features
    wine_raw.to_csv(export_path)
    
if __name__ == '__main__':
    main()