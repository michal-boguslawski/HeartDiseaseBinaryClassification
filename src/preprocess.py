import os
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_data() -> pd.DataFrame:
    file_path = Path(__file__).parent.parent.absolute().joinpath("data", "raw", "framingham.csv")
    df = pd.read_csv(file_path)
    return df


def split_data(X: pd.DataFrame, y: pd.Series | None = None, train_size: float = 0.8, random_state: int = 42):
    if y is None:
        return train_test_split(X, train_size=train_size, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
