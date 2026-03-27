# -*- coding: utf-8 -*-
"""
src/evaluation/metrics.py
--------------------------
Métriques de classification configurables pour BotOrNot.

Toutes les métriques sont calculées en une passe et retournées dans
un dict standardisé. Compatible avec le système de CV et les modèles.

Métriques disponibles :
    - roc_auc         : AUROC (insensible au threshold)
    - pr_auc          : Area under Precision-Recall curve
    - f1              : F1-score au threshold optimal
    - precision       : Précision au threshold optimal
    - recall          : Rappel au threshold optimal
    - balanced_accuracy: Accuracy équilibrée (moyenne TPR + TNR)
    - brier_score     : Calibration (0=parfait, 0.25=aléatoire)

Philosophie :
    - AUROC  : vision d'ensemble, robuste au déséquilibre
    - PR-AUC : plus informatif sur la classe minoritaire (bots)
    - F1     : critique pour la compétition (souvent la métrique de score)
    - Brier  : vérifier que les probabilités sont calibrées
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Métriques supportées
AVAILABLE_METRICS = [
    "roc_auc",
    "pr_auc",
    "f1",
    "f1_macro",
    "precision",
    "recall",
    "balanced_accuracy",
    "brier_score",
    "accuracy",
    "mcc",  # Matthews Correlation Coefficient
]

# Métriques par défaut pour la compétition
DEFAULT_METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall", "balanced_accuracy"]


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _to_arrays(
    y_true: Union[np.ndarray, pd.Series, list],
    y_prob: Union[np.ndarray, pd.Series, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Convertit les entrées en arrays NumPy propres."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Dimensions incompatibles : y_true={y_true.shape} y_prob={y_prob.shape}"
        )
    return y_true, y_prob


def find_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trouve le seuil qui maximise le F1-score."""
    from sklearn.metrics import f1_score
    best, best_thr = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 90):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best:
            best, best_thr = float(f1), float(t)
    return best_thr


# ---------------------------------------------------------------------------
# Calcul des métriques
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    metrics:   List[str] = DEFAULT_METRICS,
    threshold: Optional[float] = None,
    optimize_threshold: bool = True,
) -> Dict[str, float]:
    """
    Calcule un ensemble de métriques de classification binaire.

    Args:
        y_true             : labels vrais (0/1)
        y_prob             : probabilités de la classe positive
        metrics            : liste de métriques à calculer
        threshold          : seuil fixe (prioritaire sur optimize_threshold)
        optimize_threshold : optimiser le seuil par F1 si True

    Returns:
        dict {metric_name → valeur}
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true, y_prob = _to_arrays(y_true, y_prob)
    results: Dict[str, float] = {}

    # Métadonnées
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    results["n_samples"] = n
    results["n_pos"]     = n_pos
    results["n_neg"]     = n_neg
    results["pos_rate"]  = round(n_pos / max(n, 1), 4)

    # Seuil
    if threshold is not None:
        thr = float(threshold)
    elif optimize_threshold:
        thr = find_f1_threshold(y_true, y_prob)
    else:
        thr = 0.5

    results["threshold"] = round(thr, 4)
    y_pred = (y_prob >= thr).astype(int)

    # --- Calcul des métriques demandées ---
    for metric in metrics:
        try:
            if metric == "roc_auc":
                if n_pos == 0 or n_neg == 0:
                    val = 0.5
                else:
                    val = float(roc_auc_score(y_true, y_prob))

            elif metric == "pr_auc":
                if n_pos == 0:
                    val = 0.0
                else:
                    val = float(average_precision_score(y_true, y_prob))

            elif metric == "f1":
                val = float(f1_score(y_true, y_pred, zero_division=0))

            elif metric == "f1_macro":
                val = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

            elif metric == "precision":
                val = float(precision_score(y_true, y_pred, zero_division=0))

            elif metric == "recall":
                val = float(recall_score(y_true, y_pred, zero_division=0))

            elif metric == "balanced_accuracy":
                val = float(balanced_accuracy_score(y_true, y_pred))

            elif metric == "accuracy":
                val = float(accuracy_score(y_true, y_pred))

            elif metric == "brier_score":
                val = float(brier_score_loss(y_true, y_prob))

            elif metric == "mcc":
                val = float(matthews_corrcoef(y_true, y_pred))

            else:
                logger.warning("Métrique inconnue : '%s'", metric)
                val = float("nan")

            results[metric] = round(val, 4)

        except Exception as e:
            logger.warning("Erreur calcul '%s' : %s", metric, e)
            results[metric] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Comparaison de modèles
# ---------------------------------------------------------------------------

def compare_models(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = "roc_auc",
) -> pd.DataFrame:
    """
    Compare plusieurs modèles à partir de leurs dicts de métriques.

    Args:
        results        : {model_name → metrics_dict}
        primary_metric : métrique de tri (descendant)

    Returns:
        DataFrame trié par primary_metric
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name, **metrics}
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Réordonner — model en premier, primary_metric en second
    cols = ["model", primary_metric] + [
        c for c in df.columns if c not in ("model", primary_metric)
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=False)

    return df.reset_index(drop=True)


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    primary_metric: str = "roc_auc",
) -> None:
    """Affiche un tableau de comparaison lisible dans le terminal."""
    df = compare_models(results, primary_metric)
    if metrics:
        show_cols = ["model"] + [m for m in metrics if m in df.columns]
        df = df[show_cols]
    print("\n" + df.to_string(index=False))
    print()
