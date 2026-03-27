# -*- coding: utf-8 -*-
"""
src/features/structural.py
---------------------------
Extracteur de features structurelles — signaux techniques non-NLP.

Principe :
    Ce module est OPPORTUNISTE. Si les colonnes n'existent pas dans le
    dataset du jour J, il retourne un DataFrame vide proprement sans planter.
    C'est le profiler qui décide si ce module doit être activé.

Signaux exploités :
    ID patterns     — séquentialité, préfixe commun, densité de chiffres,
                      IDs consécutifs (création en batch)
    Source / client — diversité des clients API, source unique (bot API)
    Batch creation  — comptes créés dans la même fenêtre temporelle
    Format anomalies— longueur d'ID anormale, format non-standard,
                      champs obligatoires vides

Usage :
    from src.features.structural import extract_structural_features

    feat_df = extract_structural_features(accounts_df, posts_df)
    # Retourne DataFrame vide si les colonnes sont absentes
"""

from __future__ import annotations

import logging
import re
import string
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import AccountCols, PostCols

logger = logging.getLogger(__name__)

_PREFIX = "str_"

# Fenêtre temporelle pour la détection de création en batch (secondes)
_BATCH_WINDOW_SECONDS = 60


# ---------------------------------------------------------------------------
# Colonnes produites
# ---------------------------------------------------------------------------

STRUCTURAL_COLS = [
    # ID patterns
    "str_id_len",                  # longueur de l'account_id (en string)
    "str_id_digit_ratio",          # ratio de chiffres dans l'ID
    "str_id_n_digits",             # nombre de chiffres dans l'ID
    "str_id_is_numeric",           # ID entièrement numérique
    "str_id_sequential_score",     # score de séquentialité avec les voisins
    # Source / client
    "str_source_n_unique",         # nb de clients API distincts utilisés
    "str_source_is_single",        # 1 si toujours le même client
    "str_source_has_api",          # 1 si contient "API" dans la source
    "str_source_entropy",          # entropie de Shannon sur les sources
    # Batch creation
    "str_batch_score",             # fraction de comptes créés dans la même seconde/minute
    "str_created_same_minute",     # 1 si créé dans la même minute qu'un autre compte
    # Format anomalies
    "str_bio_punct_ratio",         # ratio de ponctuation dans la bio
    "str_has_url_in_bio",          # URL dans la bio (signal de spam)
    "str_username_all_digits",     # username 100% chiffres
    "str_username_no_vowels",      # pas de voyelles dans le username (généré ?)
    "str_missing_bio",             # bio absente
    "str_missing_location",        # localisation absente
]


def _null_row(n: int = 1) -> dict:
    return {c: np.nan for c in STRUCTURAL_COLS}


# ---------------------------------------------------------------------------
# Extraction par compte : ID patterns
# ---------------------------------------------------------------------------

def _id_features(account_id: str) -> dict:
    """Analyse l'identifiant d'un compte."""
    s      = str(account_id)
    n_dig  = sum(c.isdigit() for c in s)
    f = {
        "str_id_len":         len(s),
        "str_id_n_digits":    n_dig,
        "str_id_digit_ratio": round(n_dig / max(len(s), 1), 4),
        "str_id_is_numeric":  int(s.isdigit()),
    }
    return f


def _sequential_score(
    sorted_ids: pd.Series,
) -> pd.Series:
    """
    Calcule le score de séquentialité pour une série d'IDs numériques triés.
    Un score proche de 1 indique des IDs créés consécutivement (batch).
    """
    numeric = pd.to_numeric(sorted_ids, errors="coerce")
    if numeric.isna().mean() > 0.5:
        return pd.Series(0.0, index=sorted_ids.index)

    diffs       = numeric.diff().abs()
    median_diff = diffs.median()
    # Score : fraction de diffs ≤ 10 × la médiane (séquels proches)
    if pd.isna(median_diff) or median_diff == 0:
        return pd.Series(0.0, index=sorted_ids.index)

    score = (diffs <= max(median_diff * 10, 5)).astype(float)
    return score.fillna(0.0)


# ---------------------------------------------------------------------------
# Extraction : source / client
# ---------------------------------------------------------------------------

def _source_features(group: pd.DataFrame) -> dict:
    """Features de diversité de la source (client API)."""
    if PostCols.SOURCE not in group.columns:
        return {
            "str_source_n_unique": np.nan,
            "str_source_is_single": np.nan,
            "str_source_has_api": np.nan,
            "str_source_entropy": np.nan,
        }

    sources = group[PostCols.SOURCE].dropna().astype(str)
    if sources.empty:
        return {
            "str_source_n_unique": 0,
            "str_source_is_single": 1,
            "str_source_has_api": 0,
            "str_source_entropy": 0.0,
        }

    n_unique = int(sources.nunique())
    counts   = sources.value_counts(normalize=True)
    # Entropie
    h = float(-(counts * np.log2(counts + 1e-12)).sum())
    max_h = np.log2(n_unique) if n_unique > 1 else 1.0
    entropy = round(h / max_h, 4) if max_h > 0 else 0.0

    has_api  = int(sources.str.upper().str.contains("API|BOT|AUTO", na=False).any())

    return {
        "str_source_n_unique":  n_unique,
        "str_source_is_single": int(n_unique == 1),
        "str_source_has_api":   has_api,
        "str_source_entropy":   entropy,
    }


# ---------------------------------------------------------------------------
# Extraction : format / profil
# ---------------------------------------------------------------------------

def _profile_anomaly_features(row: pd.Series) -> dict:
    """Détecte des anomalies de format dans le profil du compte."""
    f: dict = {}

    # Bio
    bio = str(row.get(AccountCols.BIO, "")) if pd.notna(row.get(AccountCols.BIO)) else ""
    f["str_missing_bio"] = int(bio.strip() == "")
    if bio:
        punct_chars = sum(1 for c in bio if c in string.punctuation)
        f["str_bio_punct_ratio"] = round(punct_chars / max(len(bio), 1), 4)
        f["str_has_url_in_bio"]  = int(bool(re.search(r"https?://|www\.", bio)))
    else:
        f["str_bio_punct_ratio"] = 0.0
        f["str_has_url_in_bio"]  = 0

    # Localisation
    loc = row.get(AccountCols.LOCATION, "")
    f["str_missing_location"] = int(
        pd.isna(loc) or str(loc).strip() == ""
    )

    # Username
    uname = str(row.get(AccountCols.SCREEN_NAME, "")) if pd.notna(
        row.get(AccountCols.SCREEN_NAME)) else ""
    f["str_username_all_digits"] = int(uname.isdigit() and len(uname) > 0)
    vowels = set("aeiouAEIOU")
    f["str_username_no_vowels"]  = int(
        len(uname) > 0 and not any(c in vowels for c in uname)
    )

    return f


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def extract_structural_features(
    accounts_df: Optional[pd.DataFrame] = None,
    posts_df:    Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Extrait les features structurelles pour chaque compte.

    Le module est opportuniste :
        - Si aucune colonne utile n'est trouvée → retourne DataFrame vide propre
        - Si seulement certaines colonnes existent → calcule ce qui est possible
        - Ne plante jamais

    Args:
        accounts_df : DataFrame de comptes (colonnes canoniques)
        posts_df    : DataFrame de posts

    Returns:
        DataFrame avec account_id + features structurelles.
        Une ligne par compte. DataFrame vide (0 lignes) si rien n'est dispo.
    """
    id_col = AccountCols.ID

    if accounts_df is None and posts_df is None:
        logger.warning("[structural] Aucune donnée → bloc vide")
        return pd.DataFrame(columns=[id_col])

    # Rassembler tous les account_ids connus
    known_ids: list = []
    if accounts_df is not None and not accounts_df.empty and id_col in accounts_df.columns:
        known_ids = list(accounts_df[id_col].dropna().astype(str).unique())
    elif posts_df is not None and not posts_df.empty and id_col in posts_df.columns:
        known_ids = list(posts_df[id_col].dropna().astype(str).unique())

    if not known_ids:
        return pd.DataFrame(columns=[id_col])

    # Vérifier si au moins une feature structurelle est calculable
    can_compute_ids      = True   # toujours possible si on a les IDs
    can_compute_profile  = accounts_df is not None and not accounts_df.empty
    can_compute_source   = (posts_df is not None and not posts_df.empty
                            and PostCols.SOURCE in (posts_df.columns if posts_df is not None else []))
    can_compute_batch    = (accounts_df is not None
                            and AccountCols.CREATED_AT in (accounts_df.columns
                                                           if accounts_df is not None else []))

    logger.info(
        "[structural] IDs=%s, profil=%s, source=%s, batch=%s",
        can_compute_ids, can_compute_profile, can_compute_source, can_compute_batch,
    )

    # Index des accounts et posts
    acc_index = (accounts_df.set_index(id_col)
                 if can_compute_profile and id_col in accounts_df.columns
                 else pd.DataFrame())
    posts_grouped = (
        {k: v for k, v in posts_df.groupby(id_col)}
        if posts_df is not None and not posts_df.empty and id_col in posts_df.columns
        else {}
    )

    # Score de séquentialité global (calculé sur tous les IDs triés ensemble)
    id_series = pd.Series(known_ids)
    id_series_sorted = id_series.sort_values().reset_index(drop=True)
    seq_scores_series = _sequential_score(id_series_sorted)
    seq_score_map = dict(zip(id_series_sorted.values, seq_scores_series.values))

    # Détection de batch creation (comptes créés dans la même minute)
    batch_minute_map: dict = {}
    if can_compute_batch:
        try:
            created = acc_index[AccountCols.CREATED_AT].dropna()
            ts      = pd.to_datetime(created, utc=True, errors="coerce").dropna()
            minute_str = ts.dt.strftime("%Y-%m-%dT%H:%M")
            minute_counts = minute_str.value_counts()
            # Un compte est "créé en batch" si sa minute a ≥ 2 comptes
            for aid, min_str in minute_str.items():
                batch_minute_map[str(aid)] = int(minute_counts[min_str] >= 2)
        except Exception as e:
            logger.debug("[structural] batch detection erreur : %s", e)

    # Calcul global batch_score (fraction de comptes en batch parmi tous)
    if batch_minute_map:
        total_in_batch = sum(batch_minute_map.values())
        global_batch_score = round(total_in_batch / max(len(batch_minute_map), 1), 4)
    else:
        global_batch_score = np.nan

    # Calcul par compte
    rows = []
    for aid in known_ids:
        feat: dict = {id_col: aid}

        # ID features
        feat.update(_id_features(aid))
        feat["str_id_sequential_score"] = float(seq_score_map.get(aid, 0.0))

        # Profile anomalies
        if can_compute_profile and aid in acc_index.index:
            acc_row = acc_index.loc[aid]
            if isinstance(acc_row, pd.DataFrame):
                acc_row = acc_row.iloc[0]
            try:
                feat.update(_profile_anomaly_features(acc_row))
            except Exception as e:
                logger.debug("[structural] profile error '%s': %s", aid, e)
        else:
            for col in ["str_missing_bio", "str_bio_punct_ratio", "str_has_url_in_bio",
                        "str_missing_location", "str_username_all_digits", "str_username_no_vowels"]:
                feat[col] = np.nan

        # Source features
        if aid in posts_grouped:
            feat.update(_source_features(posts_grouped[aid]))
        else:
            for col in ["str_source_n_unique", "str_source_is_single",
                        "str_source_has_api", "str_source_entropy"]:
                feat[col] = np.nan

        # Batch
        feat["str_batch_score"]        = global_batch_score
        feat["str_created_same_minute"] = float(batch_minute_map.get(aid, np.nan))

        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=[id_col])

    result_df = pd.DataFrame(rows).reset_index(drop=True)

    # Typage numérique
    num_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[num_cols] = result_df[num_cols].astype(np.float32)

    logger.info(
        "[structural] %d comptes × %d features",
        len(result_df), len(result_df.columns) - 1,
    )
    return result_df
