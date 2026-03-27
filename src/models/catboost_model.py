# -*- coding: utf-8 -*-
"""
src/models/catboost_model.py
-----------------------------
CatBoost — excellent sur les features catégorielles et les petits datasets.

Points forts pour BotOrNot :
    - Gère bien les petits datasets (régularisation implicite)
    - Pas besoin de normalisation
    - Supporte les features catégorielles natives
    - Robuste au surapprentissage par défaut

Usage :
    from src.models.catboost_model import CatBoostDetector

    model = CatBoostDetector()
    model.fit(X_train, y_train, X_val, y_val)
    result = model.evaluate(X_val, y_val)
    print(result.summary())
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models._base import BotDetectorBase

logger = logging.getLogger(__name__)


class CatBoostDetector(BotDetectorBase):
    """
    Détecteur basé sur CatBoost.

    Hyperparamètres clés :
        iterations   : nb d'arbres
        learning_rate: taux d'apprentissage
        depth        : profondeur max (défaut 6)
        auto_class_weights: "Balanced" pour déséquilibre
    """

    name = "catboost"

    def _get_params(self) -> Dict[str, Any]:
        return {
            "iterations":        500,
            "learning_rate":     0.05,
            "depth":             6,
            "l2_leaf_reg":       3.0,
            "border_count":      128,
            "auto_class_weights":"Balanced",
            "early_stopping_rounds": 50,
            "random_seed":       42,
            "verbose":           0,
        }

    def _build_model(self) -> Any:
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError(
                "CatBoost non installé. Lancer : pip install catboost"
            )
        p = dict(self.params)
        return CatBoostClassifier(
            iterations          = p.get("iterations", 500),
            learning_rate       = p.get("learning_rate", 0.05),
            depth               = p.get("depth", 6),
            l2_leaf_reg         = p.get("l2_leaf_reg", 3.0),
            border_count        = p.get("border_count", 128),
            auto_class_weights  = p.get("auto_class_weights", "Balanced"),
            random_seed         = p.get("random_seed", self.random_state),
            verbose             = p.get("verbose", 0),
        )

    def _fit_model(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> None:
        early_stop = self.params.get("early_stopping_rounds", 50)

        if X_val is not None and y_val is not None:
            X_val_arr, y_val_arr = self._prepare(X_val, y_val)
            self.model_.fit(
                X_arr, y_arr,
                eval_set          = (X_val_arr, y_val_arr),
                early_stopping_rounds = early_stop,
                verbose           = 0,
            )
            logger.info("[%s] Meilleur nb d'itérations : %d",
                        self.name, self.model_.best_iteration_)
        else:
            self.model_.fit(X_arr, y_arr)

    def _feature_importances(self) -> Optional[pd.Series]:
        if self.feature_names_ is None or self.model_ is None:
            return None
        try:
            fi = self.model_.get_feature_importance()
            return pd.Series(
                fi,
                index=self.feature_names_,
                name=self.name,
            ).sort_values(ascending=False)
        except Exception:
            return None
