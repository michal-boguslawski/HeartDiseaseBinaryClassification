from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Literal, Annotated, Union


class TransformerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["Pipeline"]
    steps: list["Step"]


class ColumnTransformerElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transformer: PipelineModel | TransformerModel | Literal["drop", "passthrough"]
    columns: list[str] | Literal["categorical", "binary", "numeric"]


class ColumnTransformerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["ColumnTransformer"]
    params: dict[str, Any] = Field(default_factory=dict)
    transformers: list[ColumnTransformerElement]


class EvaluationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str]
    log_metrics: list[str]
    plot_metrics: list[str] | None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # General
    target: str
    random_state: int
    test_size: float

    # Model
    model_name: str
    model_params: dict[str, Any] = Field(default_factory=dict)

    # Feature selection
    preprocessing_steps: PipelineModel | None

    # Evaluation
    evaluation: EvaluationParams


Step = Union[
    ColumnTransformerModel,
    PipelineModel,
    TransformerModel,
]

PipelineModel.model_rebuild()
