import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence
from sklearn.metrics import roc_auc_score, accuracy_score, top_k_accuracy_score, \
    precision_score, recall_score, f1_score, classification_report, confusion_matrix, \
    log_loss


METRICS_REGISTRY = {
    "auc": roc_auc_score,
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "top_k": top_k_accuracy_score,
    "classification_report": classification_report,
    "confusion_matrix": confusion_matrix,
    "log_loss": log_loss,
    "partial_dependence": partial_dependence,
}


class Evaluator:
    def __init__(self, metrics: list[str], log_metrics: list[str], plot_metrics: list[str] | None):
        self.metrics = [METRICS_REGISTRY[metric] for metric in metrics]
        self.log_metrics = [METRICS_REGISTRY[metric] for metric in log_metrics]
        self.plot_metrics = [METRICS_REGISTRY[metric] for metric in plot_metrics] if plot_metrics else None

    def _evaluate_metrics(self, y_true, y_pred, y_proba=None) -> dict[str, float]:
        results = {}
        for metric in self.metrics:
            if metric.__name__ == "top_k":
                results[metric.__name__] = metric(y_true, y_proba, k=5)
            else:
                results[metric.__name__] = metric(y_true, y_pred)
        return results

    def evaluate(self, y_true, y_pred, y_proba=None, estimator: BaseEstimator | None = None,
                 X: pd.DataFrame | np.ndarray | None = None, features: list[str] | None = None,
                 categorical: list[str] | None = None) -> dict[str, float]:
        results = self._evaluate_metrics(y_true, y_pred, y_proba)
        return results
