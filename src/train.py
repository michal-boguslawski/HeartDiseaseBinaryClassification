from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model() -> Pipeline:
    return Pipeline([
        ('clf', LogisticRegression(
            random_state=42,
            class_weight="balanced",
            # solver="newton-cholesky",
        ))
    ])
