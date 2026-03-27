# -*- coding: utf-8 -*-
"""
src/models/lightgbm_model.py
-----------------------------
LightGBM — modèle principal du pipeline, rapide et compétitif.

Points forts pour BotOrNot :
    - Gère nativement les NaN → pas besoin d'imputation parfaite
    - Extrêmement rapide (GPU optionnel)
    - Robuste au déséquilibre de classes (scale_pos_weight)
    - Feature importance très fiable (gain-based)

Usage :
    from src.models.lightgbm_model import LightGBMDetector

    model = LightGBMDetector()
    model.fit(X_train, y_train, X_val, y_val)   # early stopping auto si X_val fourni
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


class LightGBMDetector(BotDetectorBase):
    """
    Détecteur basé sur LightGBM.

    Hyperparamètres clés :
        n_estimators    : nb d'arbres (avec early stopping → peut s'arrêter avant)
        learning_rate   : taux d'apprentissage (défaut 0.05)
        num_leaves      : complexité des arbres (défaut 63)
        scale_pos_weight: ratio neg/pos pour déséquilibre (calculé auto si None)
        early_stopping_rounds : arrêt si pas d'amélioration sur le set de validation
    """

    name = "lightgbm"

    def _get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators":          500,
            "learning_rate":         0.05,
            "num_leaves":            63,
            "max_depth":             -1,
            "min_child_samples":     20,
            "subsample":             0.8,
            "colsample_bytree":      0.8,
            "reg_alpha":             0.1,
            "reg_lambda":            0.1,
            "scale_pos_weight":      None,  # auto si None
            "early_stopping_rounds": 50,
            "verbose":               -1,
            "random_state":          42,
        }

    def _build_model(self) -> Any:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM non installé. Lancer : pip install lightgbm"
            )
        p = dict(self.params)
        # scale_pos_weight sera défini dans _fit_model si None
        return lgb.LGBMClassifier(
            n_estimators        = p.get("n_estimators", 500),
            learning_rate       = p.get("learning_rate", 0.05),
            num_leaves          = p.get("num_leaves", 63),
            max_depth           = p.get("max_depth", -1),
            min_child_samples   = p.get("min_child_samples", 20),
            subsample           = p.get("subsample", 0.8),
            colsample_bytree    = p.get("colsample_bytree", 0.8),
            reg_alpha           = p.get("reg_alpha", 0.1),
            reg_lambda          = p.get("reg_lambda", 0.1),
            class_weight        = "balanced" if p.get("scale_pos_weight") is None else None,
            random_state        = p.get("random_state", self.random_state),
            verbose             = p.get("verbose", -1),
            n_jobs              = -1,
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
            from lightgbm import early_stopping, log_evaluation
            callbacks = [
                early_stopping(early_stop, verbose=False),
                log_evaluation(period=-1),
            ]
            self.model_.fit(
                X_arr, y_arr,
                eval_set   = [(X_val_arr, y_val_arr)],
                callbacks  = callbacks,
            )
            logger.info("[%s] Meilleur nb d'arbres : %d",
                        self.name, self.model_.best_iteration_)
        else:
            self.model_.fit(X_arr, y_arr)
