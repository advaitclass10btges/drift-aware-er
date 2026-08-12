import pandas as pd
from pathlib import Path


def load_amazon_google(base_path):
    """
    Load Amazon-Google ER benchmark.

    Returns:
        table_a
        table_b
        train_pairs
        valid_pairs
        test_pairs
    """

    base_path = Path(base_path)

    table_a = pd.read_csv(base_path / "tableA.csv")
    table_b = pd.read_csv(base_path / "tableB.csv")

    train = pd.read_csv(base_path / "train.csv")
    valid = pd.read_csv(base_path / "valid.csv")
    test = pd.read_csv(base_path / "test.csv")

    return table_a, table_b, train, valid, test
