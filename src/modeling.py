from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

from src.max_minority_tree import MaxMinorityTreeClassifier


FEATURE_COLUMNS = [
    "pct_rouille",
    "rugosite",
    "saturation_moyenne",
    "indice_chlorose",
    "valeur_moyenne",
    "surface_feuille",
]


@dataclass
class ModelReport:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray


def prepare_matrix(dataframe):
    available_columns = [column for column in FEATURE_COLUMNS if column in dataframe.columns]
    X = dataframe[available_columns].to_numpy(dtype=float)
    y = dataframe["label_malade"].to_numpy(dtype=int)
    return X, y, available_columns


def _model_factories():
    return {
        "Arbre Max-Minority": lambda: MaxMinorityTreeClassifier(max_depth=3, min_samples_split=2),
        "Decision Tree (sklearn)": lambda: DecisionTreeClassifier(max_depth=3, random_state=42),
        "Random Forest (sklearn)": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    }


def evaluate_models(X: np.ndarray, y: np.ndarray):
    loo = LeaveOneOut()
    reports: list[ModelReport] = []

    for name, factory in _model_factories().items():
        y_true: list[int] = []
        y_pred: list[int] = []

        for train_idx, test_idx in loo.split(X):
            model = factory()
            model.fit(X[train_idx], y[train_idx])
            prediction = model.predict(X[test_idx])[0]
            y_true.append(int(y[test_idx][0]))
            y_pred.append(int(prediction))

        reports.append(
            ModelReport(
                name=name,
                accuracy=float(accuracy_score(y_true, y_pred)),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                confusion=confusion_matrix(y_true, y_pred),
            )
        )

    return reports


def train_final_models(X: np.ndarray, y: np.ndarray):
    trained_models = {}
    for name, factory in _model_factories().items():
        model = factory()
        model.fit(X, y)
        trained_models[name] = model
    return trained_models


def predict_label(model, X_row: np.ndarray) -> tuple[int, float | None]:
    prediction = int(model.predict(np.asarray([X_row]))[0])
    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(np.asarray([X_row]))[0]
        probability = float(np.max(proba))
    return prediction, probability