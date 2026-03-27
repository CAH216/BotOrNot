# -*- coding: utf-8 -*-
"""
src/models/xgboost_model.py
----------------------------
XGBoost — modèle competitive robuste, compatible GPU optionnel.

Points forts pour BotOrNot :
    - Mature et bien documenté
    - scale_pos_weight pour déséquilibre
    - tree_method="hist" rapide sur CPU
    - Compatible GPU (device="cuda") si disponible

Usage :
    from src.models.xgboost_model import XGBoostDetector

    model = XGBoostDetector()
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


class XGBoostDetector(BotDetectorBase):
    """
    Détecteur basé sur XGBoost.

    Hyperparamètres clés :
        n_estimators     : nb d'arbres (avec early stopping)
        learning_rate    : taux d'apprentissage (défaut 0.05)
        max_depth        : profondeur max (défaut 6)
        scale_pos_weight : ratio neg/pos (calculé auto depuis y_train si None)
        tree_method      : "hist" (rapide CPU) ou "gpu_hist" (GPU)
        early_stopping_rounds : arrêt si pas d'amélioration sur validation
    """

    name = "xgboost"

    def _get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators":          500,
            "learning_rate":         0.05,
            "max_depth":             6,
            "subsample":             0.8,
            "colsample_bytree":      0.8,
            "gamma":                 0.1,
            "reg_alpha":             0.1,
            "reg_lambda":            1.0,
            "scale_pos_weight":      None,   # auto
            "tree_method":           "hist",
            "eval_metric":           "auc",
            "early_stopping_rounds": 50,
            "random_state":          42,
            "verbosity":             0,
        }

    def _build_model(self) -> Any:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "XGBoost non installé. Lancer : pip install xgboost"
            )
        p = dict(self.params)
        return XGBClassifier(
            n_estimators        = p.get("n_estimators", 500),
            learning_rate       = p.get("learning_rate", 0.05),
            max_depth           = p.get("max_depth", 6),
            subsample           = p.get("subsample", 0.8),
            colsample_bytree    = p.get("colsample_bytree", 0.8),
            gamma               = p.get("gamma", 0.1),
            reg_alpha           = p.get("reg_alpha", 0.1),
            reg_lambda          = p.get("reg_lambda", 1.0),
            scale_pos_weight    = p.get("scale_pos_weight", 1),
            tree_method         = p.get("tree_method", "hist"),
            eval_metric         = p.get("eval_metric", "auc"),
            random_state        = p.get("random_state", self.random_state),
            verbosity           = p.get("verbosity", 0),
            n_jobs              = -1,
        )

    def _fit_model(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> None:
        p = self.params

        # Calcul auto de scale_pos_weight si None
        if p.get("scale_pos_weight") is None and y_arr is not None:
            n_neg = int((y_arr == 0).sum())
            n_pos = int((y_arr == 1).sum())
            spw   = max(1, n_neg // max(n_pos, 1))
            self.model_.set_params(scale_pos_weight=spw)
            logger.debug("[%s] scale_pos_weight auto = %d", self.name, spw)

        early_stop = p.get("early_stopping_rounds", 50)

        if X_val is not None and y_val is not None:
            X_val_arr, y_val_arr = self._prepare(X_val, y_val)
            self.model_.fit(
                X_arr, y_arr,
                eval_set              = [(X_val_arr, y_val_arr)],
                early_stopping_rounds = early_stop,
                verbose               = False,
            )
            logger.info("[%s] Meilleur nb d'arbres : %d",
                        self.name, self.model_.best_iteration)
        else:
            self.model_.fit(X_arr, y_arr)
