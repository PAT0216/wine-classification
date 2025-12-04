# download_data.py
# author: Harrison Li
# date: 2025-12-03

import click
import requests
import zipfile
from pathlib import Path

@click.command()
@click.option(
    "--pwd",
    type=str,
    help="Path to directory where raw data will be written to",
    default="~/data/raw"
)
def main(pwd):
    """Download wine quality data from UCI into the given raw-data directory."""
    raw_dir = Path(pwd).expanduser()
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    zip_path = raw_dir / "wine+quality.zip"

    if not zip_path.exists():
        print("Downloading zip from UCI...")
        resp = requests.get(url)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        target_files = {"winequality-red.csv", "winequality-white.csv"}
        existing = {p.name for p in raw_dir.glob("winequality-*.csv")}
        if target_files.issubset(existing):
            print("CSV files already exist, skipping extraction.")
        else:
            zf.extractall(raw_dir)

if __name__ == "__main__":
    main()