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


def build_preprocessing_pipeline(
    binary_features: list[str], categorical_features: list[str], numeric_features: list[str]
) -> Pipeline | ColumnTransformer:
    imputer = ColumnTransformer(
        [
            ("binary", SimpleImputer(strategy="constant", fill_value=0), ["BPMeds"]),  # based on EDA only binary variable with missing values. Distribution of missing records against Target similar to group 0
            ("categorical", SimpleImputer(strategy="constant", fill_value=0), categorical_features),  # No significat differences. Possibly merge later
            ("indicator", MissingIndicator(error_on_new=False), numeric_features),
            ("numeric", SimpleImputer(strategy="mean"), numeric_features),
        ],
        remainder = "passthrough",
        verbose_feature_names_out=False,
    )
    preprocess_pipeline = Pipeline(
        [
            ("imputer", imputer),
        ]
    )
    return imputer
