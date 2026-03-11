# Run training pipelines from here
import argparse
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn import set_config

from .config import load_config, Config
from .dataclass import PipelineModel
from .evaluate import Evaluator
from .features import get_feature_groups, PreprocessingPipelineBuilder
from .preprocess import load_data, split_data
from .train import build_model


set_config(transform_output="pandas")


def build_pipeline(preprocessing_steps: PipelineModel | None, model_name: str, model_params: dict = {}) -> Pipeline:
    preprocessing_pipeline = PreprocessingPipelineBuilder().build(preprocessing_steps)
    model = build_model(model_name, model_params)
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


def run_experiment(config: Config) -> Pipeline:
    TARGET = config.target
    RANDOM_STATE = config.random_state
    TEST_SIZE = config.test_size

    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = split_data(X, y, train_size=1-TEST_SIZE, random_state=RANDOM_STATE)

    binary_features, categorical_features, numeric_features = get_feature_groups(X_train)

    pipe = build_pipeline(config.preprocessing_steps, model_name=config.model_name, model_params=config.model_params)

    print(pipe)

    print(20*"=", "Start Train", 20*"=")
    pipe.fit(X_train, y_train)

    evaluator = Evaluator(config.evaluation.metrics, config.evaluation.log_metrics, config.evaluation.plot_metrics)

    y_pred = pipe.predict(X_train)
    y_proba = pipe.predict_proba(X_train)

    print(evaluator.evaluate(y_train, y_pred, y_proba))

    print(20*"=", "Start Test", 20*"=")
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    print(evaluator.evaluate(y_test, y_pred, y_proba))
    return pipe


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
