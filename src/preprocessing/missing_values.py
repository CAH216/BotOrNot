# -*- coding: utf-8 -*-
"""
src/preprocessing/missing_values.py
-------------------------------------
Gestion des valeurs manquantes dans le pipeline BotOrNot.

Principes :
    - Ne jamais supprimer silencieusement des lignes sans rapport
    - Stratégie par type de colonne (numérique / texte / datetime / catégorielle)
    - Laisser les colonnes inconnues telles quelles par défaut
    - Retourner un rapport de ce qui a été imputé

Fonctions publiques :
    impute_accounts_df(df)   → DataFrame comptes nettoyé
    impute_posts_df(df)      → DataFrame posts nettoyé
    fill_missing(df, strategy_map) → contrôle fin par colonne
    missing_report(df)       → analyse rapide des valeurs manquantes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.schema import AccountCols, PostCols

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rapport de valeurs manquantes
# ---------------------------------------------------------------------------

@dataclass
class MissingReport:
    """Résumé de ce qui a été imputé."""
    filled:    Dict[str, Any]   = field(default_factory=dict)  # col → valeur utilisée
    unchanged: List[str]        = field(default_factory=list)
    dropped_rows: int           = 0
    n_rows_before: int          = 0
    n_rows_after:  int          = 0

    def summary(self) -> str:
        lines = [
            f"MissingReport : {self.n_rows_before} → {self.n_rows_after} lignes",
            f"  Imputations : {len(self.filled)}",
        ]
        for col, val in self.filled.items():
            lines.append(f"    '{col}' → {repr(val)}")
        if self.unchanged:
            lines.append(f"  Inchangées : {self.unchanged}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyse rapide
# ---------------------------------------------------------------------------

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame résumant les valeurs manquantes.

    Colonnes : col, n_missing, pct_missing, dtype
    """
    records = []
    for col in df.columns:
        n = int(df[col].isna().sum())
        records.append({
            "col":         col,
            "n_missing":   n,
            "pct_missing": round(100 * n / max(len(df), 1), 2),
            "dtype":       str(df[col].dtype),
        })
    return pd.DataFrame(records).sort_values("pct_missing", ascending=False)


# ---------------------------------------------------------------------------
# Imputation générique
# ---------------------------------------------------------------------------

def fill_missing(
    df: pd.DataFrame,
    strategy_map: Dict[str, Any],
) -> tuple[pd.DataFrame, MissingReport]:
    """
    Impute les valeurs manquantes selon un mapping colonne → stratégie.

    Stratégies supportées :
        - Une valeur scalaire : 0, "", "unknown", False, ...
        - "median"  : médiane de la colonne
        - "mean"    : moyenne de la colonne
        - "mode"    : valeur la plus fréquente
        - "ffill"   : propagation forward
        - "bfill"   : propagation backward
        - "drop"    : supprimer les lignes avec NaN dans cette colonne

    Args:
        df           : DataFrame source
        strategy_map : {colonne → stratégie}

    Returns:
        (df_imputé, MissingReport)
    """
    report = MissingReport(n_rows_before=len(df))
    df = df.copy()

    for col, strategy in strategy_map.items():
        if col not in df.columns:
            continue
        if df[col].isna().sum() == 0:
            report.unchanged.append(col)
            continue

        if strategy == "drop":
            before = len(df)
            df = df.dropna(subset=[col])
            report.dropped_rows += before - len(df)
            report.filled[col] = "dropped_rows"

        elif strategy == "median":
            val = df[col].median()
            df[col] = df[col].fillna(val)
            report.filled[col] = f"median={val:.4g}"

        elif strategy == "mean":
            val = df[col].mean()
            df[col] = df[col].fillna(val)
            report.filled[col] = f"mean={val:.4g}"

        elif strategy == "mode":
            mode_vals = df[col].mode()
            val = mode_vals.iloc[0] if not mode_vals.empty else None
            if val is not None:
                df[col] = df[col].fillna(val)
                report.filled[col] = f"mode={val}"

        elif strategy == "ffill":
            df[col] = df[col].ffill()
            report.filled[col] = "ffill"

        elif strategy == "bfill":
            df[col] = df[col].bfill()
            report.filled[col] = "bfill"

        else:
            # Valeur scalaire directe
            df[col] = df[col].fillna(strategy)
            report.filled[col] = strategy

    report.n_rows_after = len(df)
    return df, report


# ---------------------------------------------------------------------------
# Imputation pour accounts_df
# ---------------------------------------------------------------------------

# Stratégie par défaut pour les colonnes de comptes
_ACCOUNT_STRATEGY: Dict[str, Any] = {
    AccountCols.BIO:           "",          # bio vide = chaîne vide
    AccountCols.LOCATION:      "",
    AccountCols.LANG:          "unknown",
    AccountCols.FOLLOWERS:     0,           # 0 si absent
    AccountCols.FOLLOWING:     0,
    AccountCols.TOTAL_POSTS:   0,
    AccountCols.VERIFIED:      False,
    AccountCols.PROFILE_IMAGE: True,        # par défaut = image par défaut
}


def impute_accounts_df(
    df: pd.DataFrame,
    extra_strategies: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, MissingReport]:
    """
    Impute les valeurs manquantes dans un DataFrame de comptes.

    Stratégies par colonne :
        bio, location, lang → "" / "unknown"
        followers, following, total_posts → 0
        verified → False
        default_profile_image → True

    Args:
        df               : DataFrame de comptes normalisé
        extra_strategies : mapping supplémentaire pour surcharger les défauts

    Returns:
        (df_imputé, MissingReport)
    """
    strategies = dict(_ACCOUNT_STRATEGY)
    if extra_strategies:
        strategies.update(extra_strategies)

    df, report = fill_missing(df, strategies)
    logger.debug("Imputation comptes : %s", report.filled)
    return df, report


# ---------------------------------------------------------------------------
# Imputation pour posts_df
# ---------------------------------------------------------------------------

# Stratégie par défaut pour les colonnes de posts
_POST_STRATEGY: Dict[str, Any] = {
    PostCols.TEXT:         "",          # texte vide au lieu de NaN
    PostCols.TEXT_CLEAN:   "",
    PostCols.SOURCE:       "unknown",
    PostCols.LANG:         "unknown",
    PostCols.IN_REPLY_TO:  None,        # None = pas une réponse
    PostCols.RETWEET_OF:   None,        # None = pas un retweet
}


def impute_posts_df(
    df: pd.DataFrame,
    extra_strategies: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, MissingReport]:
    """
    Impute les valeurs manquantes dans un DataFrame de posts.

    Stratégies par colonne :
        text, text_clean → ""
        source, lang → "unknown"
        in_reply_to, retweet_of → None (conserver NaN visuellement mais ok)

    Args:
        df               : DataFrame de posts normalisé
        extra_strategies : mapping supplémentaire

    Returns:
        (df_imputé, MissingReport)
    """
    strategies = dict(_POST_STRATEGY)
    if extra_strategies:
        strategies.update(extra_strategies)

    df, report = fill_missing(df, strategies)
    logger.debug("Imputation posts : %s", report.filled)
    return df, report


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def preprocess_missing(
    accounts_df: Optional[pd.DataFrame] = None,
    posts_df:    Optional[pd.DataFrame] = None,
    account_strategies: Optional[Dict[str, Any]] = None,
    post_strategies:    Optional[Dict[str, Any]] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], dict]:
    """
    Applique l'imputation sur accounts_df et/ou posts_df.

    Args:
        accounts_df        : DataFrame de comptes
        posts_df           : DataFrame de posts
        account_strategies : stratégies personnalisées pour les comptes
        post_strategies    : stratégies personnalisées pour les posts

    Returns:
        (accounts_df_imputé, posts_df_imputé, {reports})
    """
    reports: dict = {}

    if accounts_df is not None and not accounts_df.empty:
        accounts_df, acc_report = impute_accounts_df(accounts_df, account_strategies)
        reports["accounts"] = acc_report

    if posts_df is not None and not posts_df.empty:
        posts_df, post_report = impute_posts_df(posts_df, post_strategies)
        reports["posts"] = post_report

    return accounts_df, posts_df, reports
