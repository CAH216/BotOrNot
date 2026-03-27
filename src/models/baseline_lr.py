# -*- coding: utf-8 -*-
"""
src/models/baseline_lr.py
--------------------------
Régression logistique — baseline rapide et interprétable.

Toujours utile comme :
    - Sanity check (si LR > random forest → données trop simples ou fuite)
    - Interprétabilité des coefficients (signe = direction du signal)
    - Référence de vitesse d'entraînement

Usage :
    from src.models.baseline_lr import LogisticRegressionDetector

    model = LogisticRegressionDetector()
    model.fit(X_train, y_train, X_val, y_val)
    result = model.evaluate(X_val, y_val)
    print(result.summary())
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models._base import BotDetectorBase


class LogisticRegressionDetector(BotDetectorBase):
    """
    Régression logistique avec standardisation intégrée.

    Hyperparamètres clés :
        C            : inverse de la régularisation L2 (défaut 1.0)
        penalty      : "l2" (stable) ou "l1" (sparse / sélection auto)
        max_iter     : nb d'itérations max du solveur
        class_weight : "balanced" ou None
    """

    name = "logistic_regression"

    def _get_params(self) -> Dict[str, Any]:
        return {
            "C":            1.0,
            "penalty":      "l2",
            "solver":       "lbfgs",
            "max_iter":     1000,
            "class_weight": "balanced",
            "random_state": 42,
        }

    def _build_model(self) -> Pipeline:
        p = dict(self.params)
        # class_weight géré par le modèle lui-même
        lr = LogisticRegression(
            C            = p.get("C", 1.0),
            penalty      = p.get("penalty", "l2"),
            solver       = p.get("solver", "lbfgs"),
            max_iter     = p.get("max_iter", 1000),
            class_weight = p.get("class_weight", "balanced"),
            random_state = p.get("random_state", self.random_state),
        )
        # StandardScaler important : la LR est sensible à l'échelle
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),   # with_mean=False → sparse-safe
            ("lr",     lr),
        ])

    def _fit_model(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> None:
        """Entraînement simple — sklearn gère tout."""
        self.model_.fit(X_arr, y_arr)

    def _feature_importances(self) -> Optional[pd.Series]:
        """Coefficients de la LR (valeur absolue = importance)."""
        if self.feature_names_ is None or self.model_ is None:
            return None
        lr = self.model_.named_steps["lr"]
        if not hasattr(lr, "coef_"):
            return None
        return pd.Series(
            np.abs(lr.coef_[0]),
            index=self.feature_names_,
            name=self.name,
        ).sort_values(ascending=False)
