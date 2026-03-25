import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.dataclass import PipelineModel
from src.features import PreprocessingPipelineBuilder


MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "DummyClassifier": DummyClassifier,
    "XGBClassifier": XGBClassifier,
}


class Model:
    def __init__(self, preprocessing_steps: PipelineModel | None, model_name: str, model_params: dict = {}):
        self.preprocessing_steps = preprocessing_steps
        self.model_name = model_name
        self.model_params = model_params
        self.model = self._build_pipeline(preprocessing_steps, model_name, model_params)

    @staticmethod
    def _build_model(model_name: str, model_params: dict = {}) -> BaseEstimator:
        model = MODEL_REGISTRY[model_name]
        return model(**model_params)

    @staticmethod
    def _build_pipeline(preprocessing_steps: PipelineModel | None, model_name: str, model_params: dict = {}) -> Pipeline:
        preprocessing_pipeline = PreprocessingPipelineBuilder().build(preprocessing_steps)
        model = Model._build_model(model_name, model_params)
        if preprocessing_pipeline is None:
            return Pipeline(
                [("model", model)]
            )
        return Pipeline(
            [
                ("preprocessing", preprocessing_pipeline),
                ("model", model),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict_proba(X)
