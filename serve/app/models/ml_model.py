import mlflow
from mlflow.pyfunc import PyFuncModel
from mlflow.sklearn import load_model
import numpy as np
import pandas as pd
from shap import Explainer
from sklearn.pipeline import Pipeline

from ..config import MODEL_URI


class MLModel:
    def __init__(self):
        self.model: Pipeline | None = None
        self.feature_names: list[str] | None = None

    async def load(self):
        self.model = load_model(MODEL_URI)
        pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
        input_example = pyfunc_model.metadata.signature.inputs
        self.feature_names = [col.name for col in input_example.inputs]

    async def predict(self, data: pd.DataFrame) -> np.ndarray | tuple:
        if self.model is None:
            raise Exception("Model not loaded")
        return self.model.predict(data)

    async def predict_proba(self, data: pd.DataFrame) -> np.ndarray | tuple:
        if self.model is None:
            raise Exception("Model not loaded")
        return self.model.predict_proba(data)

    async def shap(self, data: pd.DataFrame):
        explainer = Explainer(self.model, data)
        return explainer(data)
