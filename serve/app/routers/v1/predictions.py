from fastapi import APIRouter, Depends
import numpy as np
import pandas as pd

from app.models.ml_model import MLModel
from app.models.utils import get_model
from app.schemas import HeartDiseaseInputs, PredictProbaResponse, PredictResponse, FeatureNamesResponse
from fastapi import HTTPException


router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    data: HeartDiseaseInputs,
    model: MLModel = Depends(get_model)
) -> PredictResponse:
    df = pd.DataFrame([data.model_dump()])
    result = await model.predict(df)

    if isinstance(result, np.ndarray):
        result = result.tolist()

    return PredictResponse(
        input=data,
        prediction=int(result[0]),
    )

@router.post("/predict_proba", response_model=PredictProbaResponse)
async def predict_proba(
    data: HeartDiseaseInputs,
    model: MLModel = Depends(get_model),
):
    df = pd.DataFrame([data.model_dump()])
    result = await model.predict_proba(df)

    if isinstance(result, np.ndarray):
        result = result.tolist()

    return PredictProbaResponse(
        input=data,
        prediction=float(result[0][1]),
    )

@router.get("/feature_names", response_model=FeatureNamesResponse)
async def feature_names(model: MLModel = Depends(get_model)):
    if model.feature_names is None:
        raise HTTPException(status_code=500, detail="Feature names not available")
    return FeatureNamesResponse(
        model_name="tmp",
        feature_names=model.feature_names,
    )
