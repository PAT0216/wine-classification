import click
import os
import sys
import requests
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo 

@click.command()
def main():
    """downloads data to data/raw"""
    wine_raw = fetch_ucirepo(id=186).data.features
    wine_raw.to_csv("../data/raw/wine-raw.csv")
    
if __name__ == '__main__':
    main()