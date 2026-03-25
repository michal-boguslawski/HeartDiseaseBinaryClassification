# Run training pipelines from here
import argparse
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.sklearn import autolog
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn import set_config

from .config import load_config, Config, MLFLOW_URI
from .dataclass import PipelineModel
from .evaluate import Evaluator
from .features import get_feature_groups, PreprocessingPipelineBuilder
from .preprocess import load_data, split_data
from .train import Model


mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("MLflow Heart Disease")
# Enable autologging
autolog()
set_config(transform_output="pandas")


def run_experiment(config: Config) -> Model | None:
    TARGET = config.target
    RANDOM_STATE = config.random_state
    TEST_SIZE = config.test_size

    df = load_data()
    X = df.drop(columns=[TARGET])

    binary_features, categorical_features, numeric_features = get_feature_groups(X)
    for col in categorical_features:
        X[col] = pd.Categorical(df[col].astype(str))
    
    y = df[TARGET]

    X_train, X_test, y_train, y_test = split_data(X, y, train_size=1-TEST_SIZE, random_state=RANDOM_STATE)

    model = None
    with mlflow.start_run(run_name=config.model_name) as experiment:
        model = Model(config.preprocessing_steps, model_name=config.model_name, model_params=config.model_params)

        print(model.model)

        print(20*"=", "Start Train", 20*"=")
        model.fit(X_train, y_train)

        background_data = X_train.sample(100)

        background_data.to_csv("background.csv", index=False)
        mlflow.log_artifact("background.csv", artifact_path="shap")

        evaluator = Evaluator(config.evaluation.metrics, config.evaluation.log_metrics, config.evaluation.plot_metrics)

        y_pred = model.predict(X_train)
        y_proba = model.predict_proba(X_train)

        training_score = evaluator.evaluate(y_train, y_pred, y_proba)
        print(training_score)
        mlflow.log_metrics({f"training_{key}": value for key, value in training_score.items()})

        print(20*"=", "Start Test", 20*"=")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        test_score = evaluator.evaluate(y_test, y_pred, y_proba)
        print(test_score)
        mlflow.log_metrics({f"test_{key}": value for key, value in test_score.items()})
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to config file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    pipe = run_experiment(config)
