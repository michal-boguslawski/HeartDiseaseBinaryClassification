import json
from pprint import pprint
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Binarizer, OneHotEncoder, TargetEncoder, LabelEncoder, PolynomialFeatures
from typing import Literal, Any

from .config import load_config
from .dataclass import ColumnTransformerModel, TransformerModel, PipelineModel


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


ESTIMATOR_REGISTRY = {
    "SimpleImputer": SimpleImputer,
    "TargetEncoder": TargetEncoder,
    "MissingIndicator": MissingIndicator,
    "Binarizer": Binarizer,
    "OneHotEncoder": OneHotEncoder,
    "LabelEncoder": LabelEncoder,
    "passthrough":  "passthrough",
    "drop": "drop",
    "PolynomialFeatures": PolynomialFeatures,
}


class PreprocessingPipelineBuilder:
    _BUILDER_METHODS_REGISTRY = {
        "Pipeline": "_build_pipeline",
        "ColumnTransformer": "_build_column_transformer",
    }

    def _select_method_to_build(
        self,
        config: ColumnTransformerModel | TransformerModel | PipelineModel
    ):
        method_name = self._BUILDER_METHODS_REGISTRY.get(config.name, "_build_transformer")
        method = getattr(self, method_name)
        return method(config)

    def _build_column_transformer(
        self,
        config: ColumnTransformerModel
    ) -> ColumnTransformer:
        transformers = [
            (
                transformer_config.transformer if transformer_config.transformer in ("drop", "passthrough") else self._select_method_to_build(transformer_config.transformer),
                transformer_config.columns
            )
            for transformer_config in config.transformers
        ]
        return make_column_transformer(
            *transformers,
            **config.params
        )
        
    @staticmethod
    def _build_transformer(
        config: TransformerModel
    ) -> BaseEstimator | Literal["drop", "passthrough"]:
        estimator = ESTIMATOR_REGISTRY[config.name]
        if estimator in ("drop", "passthrough"):
            return estimator
        return estimator(**config.params)

    def _build_pipeline(
        self,
        config: PipelineModel
    ) -> Pipeline:
        transformers = [self._select_method_to_build(step) for step in config.steps]
        return make_pipeline(*transformers)

    def build(self, config: PipelineModel | None) -> Pipeline | None:
        if config is None:
            return None
        return self._build_pipeline(config)


if __name__ == "__main__":
    config_path = "configs/baseline.yaml"
    config = load_config(config_path)
    pipeline = PreprocessingPipelineBuilder().build(config.preprocessing_steps)
    pprint(config)
    if pipeline:
        pprint(pipeline.get_params(deep=True), width=200, depth=None)
    # print(json.dumps(pipeline.get_params(deep=True), indent=4, default=str))
