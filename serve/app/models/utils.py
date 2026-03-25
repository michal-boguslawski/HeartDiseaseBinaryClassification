from fastapi import Request
from .ml_model import MLModel

# Create a dictionary mapping variable name -> Python type
def get_model(request: Request) -> MLModel:
    return request.app.state.ml_model
