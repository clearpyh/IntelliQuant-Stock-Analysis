from pathlib import Path
import pandas as pd

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: Path):
    ensure_dir(path.parent)
    df.to_parquet(path)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
