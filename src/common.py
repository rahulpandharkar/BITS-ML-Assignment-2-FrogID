"""Shared dataset schema and evaluation functions."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

TARGET = "Species"
GROUP = "RecordID"
MFCC_COLUMNS = [f"MFCCs_{index:02d}" for index in range(1, 23)]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the UCI CSV's inconsistent spaces in MFCC column names."""
    renamed: dict[Any, str] = {}
    for column in frame.columns:
        name = str(column).strip()
        match = re.fullmatch(r"MFCCs_\s*(\d+)", name)
        renamed[column] = f"MFCCs_{int(match.group(1)):02d}" if match else name
    return frame.rename(columns=renamed)


def mfcc_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if re.fullmatch(r"MFCCs_\d{2}", str(c))]
    return sorted(cols, key=lambda value: int(value.rsplit("_", 1)[1]))


def score_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[str] | np.ndarray,
) -> dict[str, float | None]:
    """Return the six rubric metrics using macro averages for class balance."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    observed = set(pd.Series(y_true).astype(str).unique())
    expected = set(map(str, class_labels))
    auc: float | None
    if observed == expected:
        auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=class_labels,
                multi_class="ovr",
                average="macro",
            )
        )
    else:
        auc = None

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC (macro OVR)": auc,
        "Precision (macro)": float(precision),
        "Recall (macro)": float(recall),
        "F1 (macro)": float(f1),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

