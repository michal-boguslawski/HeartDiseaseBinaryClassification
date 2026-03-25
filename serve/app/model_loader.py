
from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow
from sklearn import set_config

from .config import MLFLOW_URI
from .models.ml_model import MLModel


mlflow.set_tracking_uri(MLFLOW_URI)
set_config(transform_output="pandas")


@asynccontextmanager
async def lifespan_load_model(app: FastAPI):
    ml_model = MLModel()
    # Load model at startup
    await ml_model.load()
    app.state.ml_model = ml_model
    yield
    # Cleanup
    app.state.ml_model = None
