from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def max_minority_purity(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(np.max(counts) / y.size)


@dataclass
class TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    prediction: int | None = None
    class_counts: dict[int, int] | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class MaxMinorityTreeClassifier:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_in_: int | None = None
        self.classes_: np.ndarray | None = None
        self.root_: TreeNode | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X doit être une matrice 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X et y doivent avoir le même nombre d'échantillons")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Le modèle n'est pas encore entraîné")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None or self.classes_ is None:
            raise ValueError("Le modèle n'est pas encore entraîné")
        X = np.asarray(X, dtype=float)
        probabilities = []
        for row in X:
            node = self._traverse(row, self.root_)
            counts = node.class_counts or {}
            total = sum(counts.values()) or 1
            probabilities.append([counts.get(int(cls), 0) / total for cls in self.classes_])
        return np.asarray(probabilities, dtype=float)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        prediction = self._majority_class(y)
        class_counts = {int(cls): int(count) for cls, count in zip(*np.unique(y, return_counts=True))}
        node = TreeNode(prediction=prediction, class_counts=class_counts)

        if depth >= self.max_depth or y.size < self.min_samples_split or np.unique(y).size == 1:
            return node

        feature_index, threshold, gain = self._best_split(X, y)
        if feature_index is None or threshold is None or gain <= 0:
            return node

        mask_left = X[:, feature_index] <= threshold
        mask_right = ~mask_left
        if not np.any(mask_left) or not np.any(mask_right):
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[mask_left], y[mask_left], depth + 1)
        node.right = self._build_tree(X[mask_right], y[mask_right], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_feature = None
        best_threshold = None
        best_score = -np.inf
        parent_purity = max_minority_purity(y)

        for feature_index in range(X.shape[1]):
            values = np.unique(X[:, feature_index])
            if values.size < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                weighted_purity = (
                    (y_left.size / y.size) * max_minority_purity(y_left)
                    + (y_right.size / y.size) * max_minority_purity(y_right)
                )
                if weighted_purity > best_score:
                    best_score = weighted_purity
                    best_feature = feature_index
                    best_threshold = float(threshold)

        if best_score <= parent_purity:
            return None, None, 0.0
        return best_feature, best_threshold, best_score - parent_purity

    def _majority_class(self, y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def _traverse(self, row: np.ndarray, node: TreeNode) -> TreeNode:
        current = node
        while not current.is_leaf:
            if row[current.feature_index] <= current.threshold:
                current = current.left or current
            else:
                current = current.right or current
        return current

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> int:
        return int(self._traverse(row, node).prediction)