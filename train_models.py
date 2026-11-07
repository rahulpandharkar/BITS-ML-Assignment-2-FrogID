"""Train five classifiers and create the reproducible held-out evaluation set."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.common import GROUP, MODEL_FILES, TARGET, canonicalize_columns, mfcc_columns, score_predictions

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Frogs_MFCCs.csv"
TEST_PATH = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TEST_FRACTION = 0.20
RANDOM_STATE = 26


def grouped_stratified_holdout(
    frame: pd.DataFrame,
    target: str = TARGET,
    group: str = GROUP,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.Index, pd.Index]:
    """Choose whole recording groups, targeting the same holdout fraction per species.

    The UCI recordings each contain a single species. For every species, this
    selects a non-empty proper subset of its recordings closest to the desired
    row count. Thus every class remains represented in both train and test.
    """
    if frame.groupby(group)[target].nunique().gt(1).any():
        raise ValueError("A recording group contains multiple species; revise the split strategy.")

    held_out: list[str] = []
    for label, class_rows in frame.groupby(target, sort=True):
        sizes = class_rows.groupby(group, sort=True).size()
        groups = list(sizes.index)
        if len(groups) < 2:
            raise ValueError(f"Species {label!r} has fewer than two recording groups.")

        goal = len(class_rows) * test_fraction
        choices: list[tuple[tuple[float, int, tuple[str, ...]], tuple[str, ...]]] = []
        for count in range(1, len(groups)):
            for subset in itertools.combinations(groups, count):
                rows = int(sizes.loc[list(subset)].sum())
                stable_subset = tuple(map(str, subset))
                rank = (abs(rows - goal), count, stable_subset)
                choices.append((rank, subset))
        held_out.extend(min(choices, key=lambda item: item[0])[1])

    test_mask = frame[group].isin(held_out)
    train_indices = frame.index[~test_mask]
    test_indices = frame.index[test_mask]
    train_groups = set(frame.loc[train_indices, group])
    test_groups = set(frame.loc[test_indices, group])
    if train_groups & test_groups:
        raise AssertionError("Recording leakage detected across train and test.")
    if set(frame.loc[train_indices, target]) != set(frame[target]):
        raise AssertionError("At least one species is absent from the training split.")
    if set(frame.loc[test_indices, target]) != set(frame[target]):
        raise AssertionError("At least one species is absent from the test split.")
    return train_indices, test_indices


def make_models() -> dict[str, object]:
    """Return transparent baseline models, scaling only distance/linear models."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2500, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        SVC(C=3.0, kernel="rbf", random_state=RANDOM_STATE),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    raw = canonicalize_columns(pd.read_csv(DATA_PATH))
    features = mfcc_columns(raw)
    required = set(features + [TARGET, GROUP])
    missing = required - set(raw.columns)
    if missing or len(features) != 22:
        raise ValueError(f"Expected 22 MFCC columns plus {TARGET} and {GROUP}; missing={sorted(missing)}")
    if raw[features + [TARGET, GROUP]].isna().any().any():
        raise ValueError("The source data contains missing values in required columns.")

    train_idx, test_idx = grouped_stratified_holdout(raw)
    x_train = raw.loc[train_idx, features]
    y_train = raw.loc[train_idx, TARGET].astype(str)
    x_test = raw.loc[test_idx, features]
    y_test = raw.loc[test_idx, TARGET].astype(str)
    classes = sorted(raw[TARGET].astype(str).unique())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    test_export = x_test.copy()
    test_export[TARGET] = y_test.to_numpy()
    test_export.to_csv(TEST_PATH, index=False)

    metrics: dict[str, dict[str, float | None]] = {}
    reports: dict[str, dict] = {}
    for name, estimator in make_models().items():
        estimator.fit(x_train, y_train)
        predicted = estimator.predict(x_test)
        probabilities = estimator.predict_proba(x_test)
        metrics[name] = score_predictions(y_test, predicted, probabilities, classes)
        reports[name] = classification_report(
            y_test,
            predicted,
            labels=classes,
            target_names=classes,
            zero_division=0,
            output_dict=True,
        )
        joblib.dump(estimator, MODEL_DIR / MODEL_FILES[name], compress=3)
        print(f"Trained {name}")

    with (REPORT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (REPORT_DIR / "classification_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
    split = {
        "dataset": "UCI Anuran Calls (MFCCs)",
        "target": TARGET,
        "feature_count": len(features),
        "feature_columns": features,
        "class_count": len(classes),
        "classes": classes,
        "total_rows": len(raw),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_recordings": int(raw.loc[train_idx, GROUP].nunique()),
        "test_recordings": int(raw.loc[test_idx, GROUP].nunique()),
        "test_fraction_requested_per_class": TEST_FRACTION,
        "split_method": "deterministic species-stratified whole-recording holdout",
        "train_class_counts": {
            str(label): int(count)
            for label, count in y_train.value_counts().sort_index().items()
        },
        "test_class_counts": {
            str(label): int(count)
            for label, count in y_test.value_counts().sort_index().items()
        },
        "random_state_for_estimators": RANDOM_STATE,
        "recording_group_overlap": 0,
    }
    with (REPORT_DIR / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    print(f"\nSaved {len(MODEL_FILES)} models and test data ({len(test_idx)} rows).")
    print(pd.DataFrame(metrics).T.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

