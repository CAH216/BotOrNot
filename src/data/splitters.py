# -*- coding: utf-8 -*-
"""
src/data/splitters.py
----------------------
Stratégies de split anti-leakage pour le pipeline BotOrNot.

Problème central :
    Un compte peut avoir plusieurs lignes (posts). Si un compte se retrouve
    dans train ET dans test → fuite garantie (account-level leakage).

Stratégies disponibles :
    stratified_split      → train/val/test stratifié (un split fixe)
    group_kfold_split     → K-Fold où chaque compte reste dans un seul fold
    stratified_kfold      → Stratified K-Fold (si une ligne par compte)
    time_split            → Split temporel (passé → futur, anti-leakage temporel)
    auto_split            → Choisit automatiquement selon les données

Règle d'or :
    TOUJOURS splitter au niveau COMPTE, jamais au niveau POST.
"""

from __future__ import annotations

import logging
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

logger = logging.getLogger(__name__)

# Type pour un fold : (X_train, X_val, y_train, y_val)
FoldType = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


# ---------------------------------------------------------------------------
# Split simple train / val / test
# ---------------------------------------------------------------------------

def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """
    Split stratifié 3 voies : train / val / test.

    Si `groups` est fourni, le split se fait au niveau du groupe
    (aucun groupe ne sera dans plus d'un ensemble).

    Args:
        X            : matrice de features (index aligné avec y)
        y            : labels (0/1)
        groups       : identifiants de groupe (ex: account_id)
        val_size     : fraction de validation
        test_size    : fraction de test
        random_state : seed

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if groups is not None:
        return _group_split(X, y, groups, val_size, test_size, random_state)

    # --- Split stratifié simple ---
    # Étape 1 : séparer test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        stratify     = y,
        random_state = random_state,
    )
    # Étape 2 : séparer val depuis trainval
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size    = val_frac,
        stratify     = y_trainval,
        random_state = random_state,
    )

    _log_split(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _group_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    val_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """Split au niveau groupe pour éviter le leakage compte → post."""
    unique_groups = groups.unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_test = max(1, int(n * test_size))
    n_val  = max(1, int(n * val_size))

    test_groups  = set(unique_groups[:n_test])
    val_groups   = set(unique_groups[n_test:n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val:])

    def _mask(g_set):
        return groups.isin(g_set)

    X_train, y_train = X[_mask(train_groups)], y[_mask(train_groups)]
    X_val,   y_val   = X[_mask(val_groups)],   y[_mask(val_groups)]
    X_test,  y_test  = X[_mask(test_groups)],  y[_mask(test_groups)]

    _log_split(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _log_split(y_train, y_val, y_test):
    for name, y_s in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if len(y_s) > 0:
            pct = 100 * y_s.mean()
            logger.info("Split %s : %d lignes (%.1f%% bots)", name, len(y_s), pct)


# ---------------------------------------------------------------------------
# K-Fold générateurs
# ---------------------------------------------------------------------------

def stratified_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Generator[FoldType, None, None]:
    """
    Stratified K-Fold standard.
    À utiliser seulement si une ligne = un compte.

    Yields:
        (X_train, X_val, y_train, y_val) pour chaque fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[train_idx]
        X_v  = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_v  = y.iloc[val_idx]
        logger.info("Fold %d/%d — train=%d val=%d", fold+1, n_splits, len(X_tr), len(X_v))
        yield X_tr, X_v, y_tr, y_v


def group_kfold_split(
    X:       pd.DataFrame,
    y:       pd.Series,
    groups:  pd.Series,
    n_splits: int = 5,
    stratified: bool = True,
) -> Generator[FoldType, None, None]:
    """
    Group K-Fold : chaque compte reste dans un seul fold.

    Si `stratified=True` → StratifiedGroupKFold (maintient la proportion
    bots/humains dans chaque fold autant que possible).

    Args:
        X        : matrice de features
        y        : labels
        groups   : identifiants de groupe (account_id)
        n_splits : nb de folds
        stratified: utiliser StratifiedGroupKFold

    Yields:
        (X_train, X_val, y_train, y_val) pour chaque fold
    """
    kf_cls = StratifiedGroupKFold if stratified else GroupKFold
    kf     = kf_cls(n_splits=n_splits)

    y_arr = np.array(y)
    g_arr = np.array(groups)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_arr, g_arr)):
        X_tr = X.iloc[train_idx]
        X_v  = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_v  = y.iloc[val_idx]

        # Vérification anti-leakage
        train_groups_set = set(g_arr[train_idx])
        val_groups_set   = set(g_arr[val_idx])
        overlap = train_groups_set & val_groups_set
        if overlap:
            logger.error("LEAKAGE DÉTECTÉ : %d groupes dans train ET val !", len(overlap))
        else:
            logger.info("Fold %d/%d — %d comptes train / %d comptes val (leakage=0)",
                        fold+1, n_splits,
                        len(train_groups_set), len(val_groups_set))

        yield X_tr, X_v, y_tr, y_v


def time_split(
    X:          pd.DataFrame,
    y:          pd.Series,
    timestamps: pd.Series,
    n_splits:   int = 5,
    gap_days:   int = 7,
) -> Generator[FoldType, None, None]:
    """
    Split temporel : les données sont ordonnées dans le temps,
    chaque fold est dans le passé, la validation dans le futur.

    Vue : [──────train──────] [gap] [──val──] ...

    Args:
        X          : matrice de features
        y          : labels
        timestamps : dates de référence par ligne (ex: account created_at)
        n_splits   : nb de coupes temporelles
        gap_days   : jours de garde entre train et val (évite contamination récente)

    Yields:
        (X_train, X_val, y_train, y_val) pour chaque coupe temporelle
    """
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    sorted_idx = ts.argsort().values

    n = len(sorted_idx)
    fold_size = n // (n_splits + 1)

    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        val_start = train_end
        val_end   = min((i + 1) * fold_size, n)

        # Appliquer le gap temporel
        if gap_days > 0 and val_start > 0:
            cutoff_ts = ts.iloc[sorted_idx[train_end - 1]] + pd.Timedelta(days=gap_days)
            # Trouver le premier index val après le gap
            val_mask = ts >= cutoff_ts
            val_idx_candidates = sorted_idx[val_start:val_end]
            val_idx = [i for i in val_idx_candidates if val_mask.iloc[i]]
            train_idx = sorted_idx[:train_end]
        else:
            train_idx = sorted_idx[:train_end]
            val_idx   = sorted_idx[val_start:val_end]

        if len(val_idx) == 0:
            continue

        X_tr = X.iloc[train_idx]
        X_v  = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_v  = y.iloc[val_idx]

        logger.info(
            "TimeSplit %d/%d — train=%d val=%d (gap=%dd)",
            i, n_splits, len(X_tr), len(X_v), gap_days
        )
        yield X_tr, X_v, y_tr, y_v


# ---------------------------------------------------------------------------
# Auto-split : choisit la stratégie automatiquement
# ---------------------------------------------------------------------------

def auto_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups:     Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None,
    n_splits:   int = 5,
    val_size:   float = 0.15,
    test_size:  float = 0.15,
    mode:       str = "kfold",
) -> Generator[FoldType, None, None]:
    """
    Sélectionne automatiquement la bonne stratégie de split.

    Logique de décision :
        - Si groups fourni + mode "kfold" → GroupKFold
        - Si timestamps fourni            → TimeSplit
        - Sinon                           → StratifiedKFold

    Args:
        mode : "kfold" | "holdout" | "time"

    Yields ou Retourne les folds
    """
    if mode == "holdout":
        X_tr, X_v, X_te, y_tr, y_v, y_te = stratified_split(
            X, y, groups, val_size, test_size
        )
        yield X_tr, X_v, y_tr, y_v
        return

    if mode == "time" and timestamps is not None:
        yield from time_split(X, y, timestamps, n_splits=n_splits)
        return

    if groups is not None:
        yield from group_kfold_split(X, y, groups, n_splits=n_splits)
        return

    yield from stratified_kfold(X, y, n_splits=n_splits)
