# -*- coding: utf-8 -*-
"""
src/models/_base.py
--------------------
Classe de base commune à tous les modèles BotOrNot.

Fournit :
    - Interface unifiée : fit / predict / predict_proba / evaluate / save / load
    - Métriques standard : AUROC, F1, précision, rappel, threshold optimal
    - Gestion du déséquilibre de classes (class_weight)
    - Logging cohérent pour tous les modèles
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcule les métriques standard de classification binaire.

    Returns:
        dict : auroc, f1, precision, recall, accuracy, threshold, n_pos, n_neg
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    metrics: Dict[str, float] = {
        "n_samples": n,
        "n_pos":     n_pos,
        "n_neg":     n_neg,
        "threshold": threshold,
    }

    try:
        metrics["auroc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    except Exception:
        metrics["auroc"] = 0.0

    try:
        metrics["avg_precision"] = round(float(average_precision_score(y_true, y_prob)), 4)
    except Exception:
        metrics["avg_precision"] = 0.0

    metrics["f1"]        = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    metrics["recall"]    = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    metrics["accuracy"]  = round(float((y_true == y_pred).mean()), 4)

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Trouve le seuil optimal en maximisant la métrique donnée.

    Args:
        metric : "f1" (prudent sur faux-positifs) ou "balanced" (G-mean)
    """
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.1, 0.95, 85)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "balanced":
            tp = float(((y_pred == 1) & (y_true == 1)).sum())
            tn = float(((y_pred == 0) & (y_true == 0)).sum())
            tpr = tp / max(float(y_true.sum()), 1)
            tnr = tn / max(float((y_true == 0).sum()), 1)
            score = (tpr * tnr) ** 0.5   # G-mean
        else:
            score = float(f1_score(y_true, y_pred, zero_division=0))

        if score > best_score:
            best_score  = score
            best_thresh = float(t)

    return best_thresh


# ---------------------------------------------------------------------------
# ModelResult — conteneur de résultats standardisé
# ---------------------------------------------------------------------------

class ModelResult:
    """Résultat d'un fit() ou evaluate()."""

    def __init__(
        self,
        model_name:  str,
        metrics:     Dict[str, float],
        threshold:   float = 0.5,
        feature_importances: Optional[pd.Series] = None,
    ) -> None:
        self.model_name          = model_name
        self.metrics             = metrics
        self.threshold           = threshold
        self.feature_importances = feature_importances

    def to_dict(self) -> dict:
        d = {"model": self.model_name, **self.metrics}
        return d

    def summary(self) -> str:
        m = self.metrics
        return (
            f"[{self.model_name}] "
            f"AUROC={m.get('auroc', '?'):.4f}  "
            f"F1={m.get('f1', '?'):.4f}  "
            f"Prec={m.get('precision', '?'):.4f}  "
            f"Rec={m.get('recall', '?'):.4f}  "
            f"(thr={self.threshold:.2f})"
        )


# ---------------------------------------------------------------------------
# BotDetectorBase — interface commune
# ---------------------------------------------------------------------------

class BotDetectorBase(ABC):
    """
    Classe de base abstraite pour tous les détecteurs BotOrNot.

    Chaque sous-classe implémente :
        _build_model()  → créer l'estimateur sklearn-compatible
        _get_params()   → hyperparamètres par défaut
        name            → identifiant du modèle
    """

    name: str = "base"

    def __init__(
        self,
        params:       Optional[Dict[str, Any]] = None,
        class_weight: str = "balanced",
        threshold:    float = 0.5,
        optimize_threshold: bool = True,
        random_state: int = 42,
    ) -> None:
        self.params             = params or self._get_params()
        self.class_weight       = class_weight
        self.threshold          = threshold
        self.optimize_threshold = optimize_threshold
        self.random_state       = random_state
        self.model_             = None
        self.feature_names_     = None
        self.is_fitted_         = False
        self.train_result_      = None

    @abstractmethod
    def _get_params(self) -> Dict[str, Any]:
        """Hyperparamètres par défaut."""

    @abstractmethod
    def _build_model(self) -> Any:
        """Créer et retourner l'estimateur (non entraîné)."""

    # ------------------------------------------------------------------
    # Préparation des données
    # ------------------------------------------------------------------

    def _prepare(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convertit X, y en arrays NumPy et gère les NaN/Inf."""
        X_arr = X.values.astype(np.float32)

        # Remplacer Inf par fill value
        X_arr = np.where(np.isinf(X_arr), -999.0, X_arr)
        # NaN → -999 (signal explicite de manquant)
        X_arr = np.where(np.isnan(X_arr), -999.0, X_arr)

        y_arr = None
        if y is not None:
            y_arr = np.array(y, dtype=np.float32)
            # Supprimer les lignes avec label NaN
            valid_mask = ~np.isnan(y_arr)
            X_arr = X_arr[valid_mask]
            y_arr = y_arr[valid_mask].astype(int)

        return X_arr, y_arr

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[pd.Series]    = None,
    ) -> "BotDetectorBase":
        """
        Entraîne le modèle.

        Args:
            X_train : matrice de features (entraînement)
            y_train : labels (0/1)
            X_val   : matrice de validation (optionnel)
            y_val   : labels validation

        Returns:
            self
        """
        self.feature_names_ = list(X_train.columns)
        X_arr, y_arr = self._prepare(X_train, y_train)

        logger.info("[%s] Entraînement : %d lignes × %d features (pos=%.1f%%)",
                    self.name, len(X_arr), X_arr.shape[1],
                    100 * y_arr.mean() if y_arr is not None else 0)

        self.model_ = self._build_model()
        self._fit_model(X_arr, y_arr, X_val, y_val)
        self.is_fitted_ = True

        # Évaluation sur validation si dispo
        if X_val is not None and y_val is not None:
            result = self.evaluate(X_val, y_val)
            self.train_result_ = result
            logger.info("[%s] Val: %s", self.name, result.summary())
        else:
            # Auto-évaluation sur train (indicatif seulement)
            result = self.evaluate(X_train, y_train)
            self.train_result_ = result
            logger.info("[%s] Train: %s", self.name, result.summary())

        return self

    def _fit_model(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> None:
        """Entraînement par défaut (sklearn-compatible). Surcharger si besoin."""
        self.model_.fit(X_arr, y_arr)

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les probabilités de la classe positive (bot)."""
        self._check_fitted()
        X_arr, _ = self._prepare(X)
        proba = self.model_.predict_proba(X_arr)[:, 1]
        return proba

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Retourne les prédictions binaires (0/1)."""
        thr   = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        return (proba >= thr).astype(int)

    # ------------------------------------------------------------------
    # Évaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimize_threshold: Optional[bool] = None,
    ) -> ModelResult:
        """
        Évalue le modèle sur (X, y).
        Optimise le seuil si optimize_threshold=True.

        Returns:
            ModelResult
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        y_arr = np.array(y, dtype=int)

        # Seuil optimal en validation
        opt = self.optimize_threshold if optimize_threshold is None else optimize_threshold
        if opt:
            best_thr = find_optimal_threshold(y_arr, proba, metric="f1")
            self.threshold = best_thr
        else:
            best_thr = self.threshold

        metrics = compute_metrics(y_arr, proba, threshold=best_thr)
        fi = self._feature_importances()

        return ModelResult(
            model_name           = self.name,
            metrics              = metrics,
            threshold            = best_thr,
            feature_importances  = fi,
        )

    def _feature_importances(self) -> Optional[pd.Series]:
        """Retourne les importances si le modèle les supporte."""
        if self.feature_names_ is None or self.model_ is None:
            return None
        if hasattr(self.model_, "feature_importances_"):
            return pd.Series(
                self.model_.feature_importances_,
                index=self.feature_names_,
                name=self.name,
            ).sort_values(ascending=False)
        if hasattr(self.model_, "coef_"):
            return pd.Series(
                np.abs(self.model_.coef_[0]),
                index=self.feature_names_,
                name=self.name,
            ).sort_values(ascending=False)
        return None

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modèle en joblib + métadonnées JSON."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path.with_suffix(".joblib"))
        meta = {
            "model_name":    self.name,
            "feature_names": self.feature_names_,
            "threshold":     self.threshold,
            "params":        self.params,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("[%s] Modèle sauvegardé : %s", self.name, path)

    @classmethod
    def load(cls, path: str | Path) -> "BotDetectorBase":
        """Charge un modèle sauvegardé."""
        import joblib
        path = Path(path)
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        instance = cls(params=meta.get("params", {}))
        instance.model_         = joblib.load(path.with_suffix(".joblib"))
        instance.feature_names_ = meta.get("feature_names")
        instance.threshold      = meta.get("threshold", 0.5)
        instance.is_fitted_     = True
        return instance

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError(
                f"Le modèle [{self.name}] n'est pas entraîné. "
                "Appeler fit() d'abord."
            )
