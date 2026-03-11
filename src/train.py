from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DummyClassifier": DummyClassifier,
}


def build_model(model_name: str, model_params: dict = {}) -> BaseEstimator:
    model = MODEL_REGISTRY[model_name]
    return model(**model_params)
