"""Data loading."""
from pathlib import Path

import pandas as pd


def load_transactions(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])
    return df.sort_values(["user_id", "date"]).reset_index(drop=True)


def load_profiles(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "user_profiles.csv")


def load_labels(data_dir: Path) -> pd.DataFrame:
    """Ground truth. Only used by the evaluation, never by the pipeline."""
    return pd.read_csv(data_dir / "ground_truth_labels.csv")
