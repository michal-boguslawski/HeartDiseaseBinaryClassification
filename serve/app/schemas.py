from pydantic import BaseModel, Field, conint, confloat, field_validator
from typing import Literal, Annotated

class HeartDiseaseInputs(BaseModel):
    male: Literal['male', 'female'] = Field(..., description="Gender: male or female (will be mapped to 1/0)")
    age: Annotated[int, conint(ge=0, le=120)] = Field(..., description="Age in years, must be 0-120")
    education: Literal['1', '2', '3', '4'] = Field("1", description="Education level: 1-4")
    
    currentSmoker: Literal['yes', 'no'] = Field(..., description="'yes' if currently smoker, 'no' otherwise")
    cigsPerDay: Annotated[int, confloat(ge=0, le=120)] = Field(..., description="Number of cigarettes per day, must be 0-120")
    BPMeds: Annotated[float, confloat(ge=0)] = Field(..., description="Blood pressure medications, must be >=0")
    
    prevalentStroke: Literal['yes', 'no'] = Field(..., description="Stroke history: 'yes' or 'no'")
    prevalentHyp: Literal['yes', 'no'] = Field(..., description="Hypertension history: 'yes' or 'no'")
    diabetes: Literal['yes', 'no'] = Field(..., description="Diabetes: 'yes' or 'no'")
    
    totChol: Annotated[float, confloat(ge=0)] = Field(..., description="Total cholesterol, must be >=0")
    sysBP: Annotated[float, confloat(ge=0)] = Field(..., description="Systolic BP, must be >=0")
    diaBP: Annotated[float, confloat(ge=0)] = Field(..., description="Diastolic BP, must be >=0")
    BMI: Annotated[float, confloat(ge=0)] = Field(..., description="Body Mass Index, must be >=0")
    heartRate: Annotated[float, confloat(ge=0)] = Field(..., description="Heart rate, must be >=0")
    glucose: Annotated[float, confloat(ge=0)] = Field(..., description="Glucose level, must be >=0")

    # ------------------------
    # Validators to map to 0/1 internally
    # ------------------------
    @field_validator('male')
    def map_gender_to_int(cls, v):
        return 1 if v.lower() == 'male' else 0

    @field_validator('education')
    def map_education_to_str(cls, v):
        return float(v)

    @field_validator('currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes')
    def map_yes_no_to_int(cls, v):
        return 1 if v.lower() == 'yes' else 0

    @field_validator('cigsPerDay')
    def zero_cigs_if_not_smoker(cls, v, info):
        # Automatically set cigsPerDay = 0 if currentSmoker = 0
        if info.data.get('currentSmoker') == 0:
            return 0
        return v


class PredictProbaResponse(BaseModel):
    input: HeartDiseaseInputs
    prediction: float


class PredictResponse(BaseModel):
    input: HeartDiseaseInputs
    prediction: int


class FeatureNamesResponse(BaseModel):
    model_name: str
    feature_names: list[str]
