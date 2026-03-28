# -*- coding: utf-8 -*-
"""
src/features/coordination.py
-----------------------------
Extracteur de features de coordination légère (V1.5).

Ne recourt pas à un GNN. Grouppe les comptes par fenêtres temporelles régulières
(time bins) et calcule des métriques de synchronisation (co-occurrence de textes,
hashtags, URLs et comportements en rafale) par rapport aux autres comptes
actifs dans la même fenêtre.

Approche 100% vectorisée via Pandas pour garantir une très haute performance (O(N)).

Features produites :
    coord_text_sim         : Proportion de mots utilisés par le compte ET par d'autres comptes dans le même bin
    coord_hashtag_sync     : Proportion de hashtags partagés avec d'autres
    coord_url_sync         : Proportion d'URLs partagées avec d'autres
    coord_burst_sync       : Fréquence des rafales (>3 posts) synchronisées avec d'autres rafales (>3 comptes)
    coord_activation_score : Nombre moyen d'autres comptes actifs en même temps
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from src.data.schema import AccountCols, PostCols

logger = logging.getLogger(__name__)

COORDINATION_COLS = [
    "coord_text_sim",
    "coord_hashtag_sync",
    "coord_url_sync",
    "coord_burst_sync",
    "coord_activation_score",
]


def _explode_and_score_shared(
    df: pd.DataFrame,
    col_name: str,
    time_col: str = "time_bin",
    id_col: str = AccountCols.ID
) -> pd.Series:
    """
    1. Explode le dataframe sur `col_name` (une liste).
    2. Dédoublonne pour avoir (time_bin, account_id, item) uniques.
    3. Compte les account_ids uniques par (time_bin, item).
    4. Marque comme partagé si freq > 1.
    5. Retourne le ratio moyen de partage par (account_id, time_bin).
    """
    if col_name not in df.columns:
        return pd.Series(dtype=float)

    # Explode ignores NaN and turns lists into rows
    exploded = df[[time_col, id_col, col_name]].explode(col_name).dropna(subset=[col_name])
    if exploded.empty:
        return pd.Series(dtype=float)

    # Convertir en string pour éviter les erreurs de groupby (si URL ou autre)
    exploded[col_name] = exploded[col_name].astype(str).str.lower().str.strip()
    # Filtrer les chaînes vides
    exploded = exploded[exploded[col_name] != ""]

    # Dédoublonner : un compte utilise un item au moins une fois par bin
    unique_usage = exploded.drop_duplicates(subset=[time_col, id_col, col_name]).copy()
    if unique_usage.empty:
        return pd.Series(dtype=float)

    # Fréquence des utilisateurs par (time_bin, item)
    item_user_freq = unique_usage.groupby([time_col, col_name])[id_col].transform("count")
    
    unique_usage["is_shared"] = (item_user_freq > 1).astype(float)

    # Moyenne de partage pour chaque (account_id, time_bin)
    # Ex: Si le compte a utilisé 4 mots, et 3 sont partagés -> 0.75
    shared_ratio = unique_usage.groupby([id_col, time_col])["is_shared"].mean()
    return shared_ratio


def extract_coordination_features(
    accounts_df: Optional[pd.DataFrame],
    posts_df: Optional[pd.DataFrame],
    cfg: Optional[dict] = None
) -> pd.DataFrame:
    """
    Extrait les features de coordination temporelle pour chaque compte.
    Nécessite cfg["coordination"]["enabled"] == True pour s'exécuter.
    """
    id_col = AccountCols.ID

    # Configuration et activation
    coord_cfg = (cfg or {}).get("coordination", {})
    is_enabled = bool(coord_cfg.get("enabled", False))
    window_min = int(coord_cfg.get("time_window_minutes", 60))
    min_users  = int(coord_cfg.get("min_users_per_bin", 2))

    if not is_enabled:
        logger.info("[coordination] Module désactivé via config.")
        return pd.DataFrame(columns=[id_col])

    # Validation des données
    if posts_df is None or posts_df.empty or PostCols.CREATED_AT not in posts_df.columns:
        logger.info("[coordination] absents/pas de timestamps → stop.")
        return pd.DataFrame(columns=[id_col])

    if id_col not in posts_df.columns:
        logger.warning("[coordination] colonne id absente → stop.")
        return pd.DataFrame(columns=[id_col])

    # Préparation
    df = posts_df.copy()
    
    try:
        # Conversion UTC et filtrage NaT
        df[PostCols.CREATED_AT] = pd.to_datetime(df[PostCols.CREATED_AT], utc=True, errors="coerce")
    except Exception as e:
        logger.warning("[coordination] Erreur conversion datetime : %s", e)
        return pd.DataFrame(columns=[id_col])

    df = df.dropna(subset=[PostCols.CREATED_AT])
    if df.empty:
        return pd.DataFrame(columns=[id_col])

    # 1. Création des fenêtres temporelles (time bins)
    # round/floor selon window_min
    time_bin = df[PostCols.CREATED_AT].dt.floor(f"{window_min}min")
    df["time_bin"] = time_bin

    # 2. Préparation des colonnes texte, urls, hashtags si présentes
    if PostCols.TEXT in df.columns:
        # Mots de >= 4 lettres (pour éviter the, a, de, le, etc.)
        df["_words"] = df[PostCols.TEXT].astype(str).str.findall(r"\b[a-zA-Z]{4,}\b")
    else:
        df["_words"] = [[]] * len(df)

    if PostCols.URLS not in df.columns:
        df[PostCols.URLS] = [[]] * len(df)
        
    if PostCols.HASHTAGS not in df.columns:
        df[PostCols.HASHTAGS] = [[]] * len(df)

    # 3. Calcul relationnels vectorisés
    # -- Text (mots) --
    txt_sync = _explode_and_score_shared(df, "_words")
    # -- Hashtags --
    ht_sync = _explode_and_score_shared(df, PostCols.HASHTAGS)
    # -- URLs --
    url_sync = _explode_and_score_shared(df, PostCols.URLS)

    # -- Base d'activité (nb posts par utilisateur par bin) --
    user_bin_counts = df.groupby([id_col, "time_bin"]).size()

    # Alignons tout sur un index (account_id, time_bin)
    metrics_per_bin = pd.DataFrame(index=user_bin_counts.index)
    
    metrics_per_bin["coord_text_sim"]     = txt_sync
    metrics_per_bin["coord_hashtag_sync"] = ht_sync
    metrics_per_bin["coord_url_sync"]     = url_sync
    
    # Remplir par 0 si pas de mots, pas d'URLs, etc. (pour les calculs de moyenne)
    metrics_per_bin.fillna(0.0, inplace=True)

    # -- Bursts --
    metrics_per_bin["n_posts"] = user_bin_counts
    metrics_per_bin["is_burst"] = (user_bin_counts >= 3)
    
    # Nombre de bursts globaux dans ce bin
    bursts_in_bin = metrics_per_bin.groupby("time_bin")["is_burst"].transform("sum")
    # Burst sync: Le compte est en burst ET >= 2 AUTRES comptes sont en burst aussi
    metrics_per_bin["coord_burst_sync"] = (metrics_per_bin["is_burst"] & (bursts_in_bin >= 3)).astype(float)

    # -- Co-activation --
    # Nombre total d'utilisateurs distincts par bin
    users_in_bin = metrics_per_bin.groupby("time_bin").transform("size")
    # Actifs partagés (nous excluons le compte lui-même)
    metrics_per_bin["coord_activation_score"] = (users_in_bin - 1).clip(lower=0).astype(float)

    # 4. Filtrage : on ne calcule la coordination que sur les bins avec >= min_users
    # car s'il est tout seul, c'est forcément 0 et ça pénalise la moyenne pour rien ?
    # Ou au contraire, s'il est souvent seul, ça montre un comportement non coordonné.
    # Gardons tous les bins, mais son activation score et sync sera de 0.

    # 5. Agrégation finale par compte (moyenne de l'activité du compte à travers tous ses bins)
    final_agg = metrics_per_bin.groupby(id_col)[COORDINATION_COLS].mean().reset_index()

    # Remplir toutes les colonnes V1.5
    for col in COORDINATION_COLS:
        if col not in final_agg.columns:
            final_agg[col] = np.nan

    # Typage numérique
    num_cols = final_agg.select_dtypes(include=[np.number]).columns
    final_agg[num_cols] = final_agg[num_cols].astype(np.float32)

    logger.info(
        "[coordination] %d comptes × %d features (window=%d min)",
        len(final_agg), len(final_agg.columns) - 1, window_min
    )
    return final_agg
