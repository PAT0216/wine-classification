# download_data.py
# author: Harrison Li
# date: 2025-12-03

import click
import pandas as pd
from ucimlrepo import fetch_ucirepo 

@click.command()
@click.option('--path', type=str, 
              help="Path to directory where raw data will be written to", default='~/data/raw')
def main(path):
    """download data to data/raw"""
    wine_raw = fetch_ucirepo(id=186)    
    wine_raw.data.original.to_csv(path + "/wine-raw.csv", index=False)

if __name__ == '__main__':
    main()