import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, OneHotEncoder, TargetEncoder


def get_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    features_df = df.columns.tolist()
    binary_features_df = df.select_dtypes(include=["bool"]).columns.tolist()
    numeric_features_df = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features_df = df.select_dtypes(include=["object"]).columns.tolist()

    binary_features_default = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
    categorical_features_default = ["education", ]
    numeric_features_default = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

    binary_features = list(
        (set(features_df) & set(binary_features_default) ) |
        ( set(binary_features_df) - set(binary_features_default + categorical_features_default + numeric_features_df) )
    )

    categorical_features = list(
        ( set(features_df) & set(categorical_features_default) ) |
        ( set(categorical_features_df) - set(binary_features_default + categorical_features_default + numeric_features_df) )
    )

    numeric_features = list(
        ( set(features_df) & set(numeric_features_default) ) |
        ( set(numeric_features_df) - set(binary_features_default + categorical_features_default + numeric_features_df) )
    )

    return binary_features, categorical_features, numeric_features


def build_feature_pipeline(
    binary_features: list[str], categorical_features: list[str], numeric_features: list[str]
) -> Pipeline | ColumnTransformer:
    column_features = ColumnTransformer(
        transformers=[
            ("no_transforms", "passthrough", 
             [
                "male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", 
                'missingindicator_totChol',
                'missingindicator_glucose',
                'missingindicator_heartRate',
                'missingindicator_cigsPerDay',
                'missingindicator_BMI',
                "age", 
                # "BMI", "heartRate", "glucose",
            ]),
            ("one_hot", OneHotEncoder(drop="first", sparse_output=False,), ["education"]),
            ("target_encoder", TargetEncoder(random_state=42), ["education"]),
            ("bmi_binarizer", Binarizer(threshold=16.), ["BMI"]),
            ("heartRate_binarizer", Binarizer(threshold=40.), ["heartRate"]),
            ("glucose_binarizer", Binarizer(threshold=120.), ["glucose"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return column_features
