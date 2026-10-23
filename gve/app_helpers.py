"""Small helpers shared by the app and the eval harness."""
import pandas as pd


def month_label(m: str) -> str:
    return pd.Timestamp(m + "-01").strftime("%B %Y")
