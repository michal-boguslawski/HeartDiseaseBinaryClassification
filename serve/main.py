from fastapi import FastAPI

from app.model_loader import lifespan_load_model
from app.routers.v1.predictions import router as predictions_router


app = FastAPI(title="ML Model API", lifespan=lifespan_load_model)

app.include_router(predictions_router, prefix="/api/v1")
