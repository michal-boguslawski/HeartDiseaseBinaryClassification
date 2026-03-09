# Run training pipelines from here
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn import set_config

from .preprocess import load_data, split_data, build_preprocessing_pipeline
from .features import build_feature_pipeline, get_feature_groups
from .train import build_model


set_config(transform_output="pandas")
TARGET = "TenYearCHD"


def run_experiment() -> Pipeline:
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = split_data(X, y)

    binary_features, categorical_features, numeric_features = get_feature_groups(X_train)

    preprocessing_pipeline = build_preprocessing_pipeline(
        binary_features=binary_features,
        categorical_features=categorical_features,
        numeric_features=numeric_features
    )

    feature_engineering_pipeline = build_feature_pipeline(
        binary_features=binary_features,
        categorical_features=categorical_features,
        numeric_features=numeric_features
    )

    model = build_model()

    pipe = Pipeline(
        [
            ("preprocess", preprocessing_pipeline),
            ("feature_engineer", feature_engineering_pipeline),
            ("model", model)
        ]
    )
    pipe.fit(X_train, y_train)

    print(20*"=", "Start Train", 20*"=")
    print(f"Train Accuracy: {pipe.score(X_train, y_train)}")
    y_pred = pipe.predict(X_train)
    y_proba = pipe.predict_proba(X_train)
    print(classification_report(y_train, y_pred))
    print(f"F1-score {f1_score(y_train, y_pred)}")
    print(f"AUC-score {roc_auc_score(y_train, y_proba[:, 1])}")

    print(20*"=", "Start Test", 20*"=")
    print(f"Test Accuracy: {pipe.score(X_test, y_test)}")
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    print(classification_report(y_test, y_pred))
    print(f"F1-score {f1_score(y_test, y_pred)}")
    print(f"AUC-score {roc_auc_score(y_test, y_proba[:, 1])}")
    return pipe


if __name__ == "__main__":
    pipe = run_experiment()
    print(type(pipe))
