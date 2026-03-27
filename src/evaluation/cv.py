# -*- coding: utf-8 -*-
"""
src/evaluation/cv.py
---------------------
Pipeline de cross-validation anti-leakage pour BotOrNot.

Ce module orchestre l'entraînement et l'évaluation de modèles sur plusieurs
folds, en s'appuyant sur les splitters (src/data/splitters.py) et les
métriques (src/evaluation/metrics.py).

Design :
    - Un objet CVRunner reçoit un modèle (BotDetectorBase-compatible) et
      une stratégie de split, puis exécute les K folds.
    - Les résultats par fold + les agrégats sont retournés dans un CVResult.
    - Le modèle final peut être re-entraîné sur tout le dataset.

Usage :
    from src.evaluation.cv import CVRunner
    from src.models.lightgbm_model import LightGBMDetector

    runner = CVRunner(
        model_cls   = LightGBMDetector,
        split_mode  = "group",
        n_splits    = 5,
        metrics     = ["roc_auc", "f1", "pr_auc"],
    )
    cv_result = runner.run(X, y, groups=account_ids)
    print(cv_result.summary())
    runner.fit_final(X, y)   # modèle sur tout le dataset
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from src.data.splitters import auto_split
from src.evaluation.metrics import DEFAULT_METRICS, compute_metrics, compare_models
from src.models._base import BotDetectorBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CVResult — conteneur des résultats
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Résultats d'une validation croisée."""
    model_name:   str
    n_splits:     int
    split_mode:   str
    metrics_list: List[str]

    fold_results: List[Dict[str, float]] = field(default_factory=list)
    mean_metrics: Dict[str, float]       = field(default_factory=dict)
    std_metrics:  Dict[str, float]       = field(default_factory=dict)
    elapsed_s:    float                  = 0.0

    def add_fold(self, metrics: Dict[str, float]) -> None:
        self.fold_results.append(metrics)

    def aggregate(self) -> None:
        """Calcule moyenne et écart-type des métriques sur les folds."""
        if not self.fold_results:
            return
        df = pd.DataFrame(self.fold_results)
        num_cols = [c for c in df.columns if c not in ("n_samples", "n_pos", "n_neg", "threshold")]
        for col in num_cols:
            if col in df:
                self.mean_metrics[col] = round(float(df[col].mean()), 4)
                self.std_metrics[col]  = round(float(df[col].std()), 4)

    def summary(self, primary: str = "roc_auc") -> str:
        lines = [
            f"\n{'='*60}",
            f"  CV [{self.model_name}] — {self.n_splits}-fold {self.split_mode}",
            f"  Temps total : {self.elapsed_s:.1f}s",
            f"{'='*60}",
        ]
        # Métriques principales
        for m in self.metrics_list:
            if m in self.mean_metrics:
                mean = self.mean_metrics[m]
                std  = self.std_metrics.get(m, 0.0)
                marker = " ◀" if m == primary else ""
                lines.append(f"  {m:<22} {mean:.4f} ± {std:.4f}{marker}")

        # Par fold
        lines.append(f"\n  Détail par fold ({primary}) :")
        for i, fold in enumerate(self.fold_results):
            val = fold.get(primary, float("nan"))
            thr = fold.get("threshold", 0.5)
            lines.append(f"    Fold {i+1} : {val:.4f}  (thr={thr:.2f})")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Retourne les résultats par fold comme DataFrame."""
        rows = []
        for i, fold in enumerate(self.fold_results):
            row = {"fold": i + 1, **fold}
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CVRunner — orchestrateur principal
# ---------------------------------------------------------------------------

class CVRunner:
    """
    Orchestrateur de cross-validation.

    Args:
        model_cls   : classe du modèle (sous-classe de BotDetectorBase)
        model_params: paramètres à passer au constructeur du modèle
        split_mode  : "stratified" | "group" | "time" | "holdout"
        n_splits    : nombre de folds
        metrics     : métriques à calculer
        optimize_threshold: optimiser le seuil F1 à chaque fold
        random_state: seed
    """

    def __init__(
        self,
        model_cls:   Type[BotDetectorBase],
        model_params: Optional[Dict[str, Any]] = None,
        split_mode:   str = "group",
        n_splits:     int = 5,
        metrics:      Optional[List[str]] = None,
        optimize_threshold: bool = True,
        random_state: int = 42,
    ) -> None:
        self.model_cls          = model_cls
        self.model_params       = model_params or {}
        self.split_mode         = split_mode
        self.n_splits           = n_splits
        self.metrics            = metrics or DEFAULT_METRICS
        self.optimize_threshold = optimize_threshold
        self.random_state       = random_state
        self.final_model_:  Optional[BotDetectorBase] = None

    def run(
        self,
        X:          pd.DataFrame,
        y:          pd.Series,
        groups:     Optional[pd.Series] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> CVResult:
        """
        Exécute la cross-validation complète.

        ANTI-LEAKAGE :
            - Si groups fourni → aucun compte dans 2 folds
            - Si split_mode="time" + timestamps → passé entraîne, futur évalue
            - Chaque fold instancie un nouveau modèle (pas de réutilisation)

        Args:
            X          : matrice de features
            y          : labels
            groups     : identifiants de groupe (account_id)
            timestamps : dates par ligne (pour split temporel)

        Returns:
            CVResult avec résultats par fold + agrégats
        """
        model_name = self.model_cls.name
        cv_result  = CVResult(
            model_name   = model_name,
            n_splits     = self.n_splits,
            split_mode   = self.split_mode,
            metrics_list = self.metrics,
        )

        logger.info("[CV] Démarrage : %s — %d folds, mode=%s",
                    model_name, self.n_splits, self.split_mode)

        t_start = time.time()
        fold_num = 0

        for X_train, X_val, y_train, y_val in auto_split(
            X, y,
            groups     = groups,
            timestamps = timestamps,
            n_splits   = self.n_splits,
            mode       = self._map_split_mode(),
        ):
            fold_num += 1
            logger.info("[CV] Fold %d — train=%d val=%d", fold_num, len(X_train), len(X_val))

            # Nouveau modèle à chaque fold (pas de fuite via état)
            model = self.model_cls(
                params            = self.model_params or None,
                optimize_threshold= False,   # on gère ici
            )

            try:
                model.fit(X_train, y_train, X_val, y_val)
                y_prob = model.predict_proba(X_val)

                fold_metrics = compute_metrics(
                    y_val, y_prob,
                    metrics            = self.metrics,
                    optimize_threshold = self.optimize_threshold,
                )
                cv_result.add_fold(fold_metrics)

                logger.info(
                    "[CV] Fold %d → roc_auc=%.4f f1=%.4f",
                    fold_num,
                    fold_metrics.get("roc_auc", float("nan")),
                    fold_metrics.get("f1", float("nan")),
                )

            except Exception as e:
                logger.error("[CV] Fold %d ÉCHOUÉ : %s", fold_num, e)
                cv_result.add_fold({m: float("nan") for m in self.metrics})

        cv_result.elapsed_s = round(time.time() - t_start, 2)
        cv_result.aggregate()
        logger.info("[CV] Terminé en %.1fs. %s", cv_result.elapsed_s, cv_result.summary())

        return cv_result

    def fit_final(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series]    = None,
    ) -> BotDetectorBase:
        """
        Re-entraîne le modèle final sur tout le dataset (train + val).
        Ce modèle est utilisé pour la prédiction finale en compétition.

        Args:
            X     : tout X (train + val fusionnés recommandé)
            y     : tous les labels
            X_val : optionnel, jeu de validation pour early stopping
            y_val : optionnel

        Returns:
            Modèle entraîné sur tout le dataset
        """
        logger.info("[CV] Entraînement du modèle final sur %d lignes", len(X))
        model = self.model_cls(params=self.model_params or None)
        model.fit(X, y, X_val, y_val)
        self.final_model_ = model
        return model

    def _map_split_mode(self) -> str:
        """Traduit le mode de split vers la convention de auto_split()."""
        mapping = {
            "stratified": "kfold",
            "group":      "kfold",
            "time":       "time",
            "holdout":    "holdout",
        }
        return mapping.get(self.split_mode, "kfold")


# ---------------------------------------------------------------------------
# Fonction de convenance : comparer plusieurs modèles en CV
# ---------------------------------------------------------------------------

def compare_models_cv(
    model_classes: List[Type[BotDetectorBase]],
    X: pd.DataFrame,
    y: pd.Series,
    groups:     Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None,
    split_mode: str = "group",
    n_splits:   int = 5,
    metrics:    Optional[List[str]] = None,
    primary_metric: str = "roc_auc",
) -> pd.DataFrame:
    """
    Exécute une CV pour chaque modèle et retourne un tableau comparatif.

    Args:
        model_classes  : liste de classes de modèles
        X, y, groups   : données
        split_mode     : stratégie de split
        n_splits       : nb de folds
        metrics        : métriques à calculer
        primary_metric : métrique de tri principal

    Returns:
        DataFrame comparatif trié par primary_metric
    """
    all_results: Dict[str, Dict[str, float]] = {}

    for cls in model_classes:
        runner     = CVRunner(cls, split_mode=split_mode, n_splits=n_splits, metrics=metrics)
        cv_result  = runner.run(X, y, groups=groups, timestamps=timestamps)
        all_results[cls.name] = cv_result.mean_metrics
        logger.info("=== %s ===\n%s", cls.name, cv_result.summary())

    return compare_models(all_results, primary_metric=primary_metric)
