import pandas as pd


def subset(df: pd.DataFrame, level: str, index: list):
    """subsetting multi-index dataframe by indexvalue of a given index.level."""
    return pd.concat([d for algo, d in df.groupby(level=level)
                      if algo in index], axis=0)
