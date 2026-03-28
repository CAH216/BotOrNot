# -*- coding: utf-8 -*-
"""
src/features/structural.py
---------------------------
Extracteur de features structurelles — V1 stable + V1.1 expérimental.

V1 (stable, toujours actif) :
    ID patterns, source/client, batch creation, format anomalies.

V1.1 (derrière 4 flags features.yaml — RULES.md §2) :
    F1 source_v11  — répétition source / homogénéité extrême
    F2 batch_v11   — batch heure/jour, densité inter-comptes
    F3 profile_v11 — cohérence username ↔ bio ↔ id
    F4 template_v11— flags "profile template-like"

Usage :
    from src.features.structural import extract_structural_features

    # V1 seule (comportement identique à avant)
    feat_df = extract_structural_features(accounts_df, posts_df)

    # V1 + familles expérimentales (nécessite validation benchmark)
    feat_df = extract_structural_features(accounts_df, posts_df, cfg={
        "structural": {
            "source_v11_enabled":   True,
            "batch_v11_enabled":    True,
            "profile_v11_enabled":  True,
            "template_v11_enabled": True,
        }
    })
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

# ---------------------------------------------------------------------------
# V1.1 — Colonnes expérimentales (derrière flags features.yaml)
# Désactivées par défaut — RULES.md §2 + §4
# ---------------------------------------------------------------------------

STRUCTURAL_COLS_V11 = {
    # F1 : source / client répétitif
    "source_v11": [
        "str_top_source_ratio",      # ratio du client le plus fréquent
        "str_source_repeat_score",   # (n_top / n_total) — biais source
        "str_source_homogeneity",    # 1 − entropie normalisée
        "str_source_is_extreme",     # 1 si top_source_ratio ≥ 0.95
    ],
    # F2 : batch creation fin (heure / jour / densité)
    "batch_v11": [
        "str_batch_same_hour",       # comptes créés dans la même heure
        "str_batch_same_day",        # comptes créés le même jour
        "str_batch_density",         # nb comptes dans la même heure / total
    ],
    # F3 : cohérence username / bio / id
    "profile_v11": [
        "str_username_bio_len_ratio",    # len(username) / len(bio) (valeur 0 si bio vide)
        "str_username_numeric_suffix",   # username se termine par chiffre(s)
        "str_id_username_len_ratio",     # len(id_str) / len(username)
        "str_profile_completeness",      # score 0-1 : % de champs non-vides
    ],
    # F4 : profil "template-like"
    "template_v11": [
        "str_bio_is_template",       # bio générique / très courte (≤ 10 car)
        "str_username_underscores",  # nb de underscores dans username
        "str_username_len",          # longueur du username
        "str_username_digits_ratio", # ratio chiffres dans username
        "str_profile_template_score",# score composite 0-4
    ],
}


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
# V1.1 — Fonctions de features expérimentales (derrière flags)
# ---------------------------------------------------------------------------

# F1 — Répétition source / homogénéité extrême
def _source_features_v11(group: pd.DataFrame) -> dict:
    """V1.1 F1 : ratio et homogénéité de la source dominante."""
    if PostCols.SOURCE not in group.columns:
        return {k: np.nan for k in STRUCTURAL_COLS_V11["source_v11"]}
    sources = group[PostCols.SOURCE].dropna().astype(str)
    if sources.empty:
        return {
            "str_top_source_ratio":    1.0,
            "str_source_repeat_score": 1.0,
            "str_source_homogeneity":  1.0,
            "str_source_is_extreme":   1,
        }
    top_ratio = float(sources.value_counts(normalize=True).iloc[0])
    n_unique  = int(sources.nunique())
    # Homogénéité = 1 − entropie normalisée
    counts = sources.value_counts(normalize=True)
    h      = float(-(counts * np.log2(counts + 1e-12)).sum())
    max_h  = np.log2(n_unique) if n_unique > 1 else 1.0
    homogeneity = round(1.0 - (h / max_h if max_h > 0 else 0.0), 4)
    return {
        "str_top_source_ratio":    round(top_ratio, 4),
        "str_source_repeat_score": round(top_ratio, 4),  # alias sémantique
        "str_source_homogeneity":  homogeneity,
        "str_source_is_extreme":   int(top_ratio >= 0.95),
    }


# F2 — Batch creation fine (heure / jour / densité)
def _batch_features_v11_build_maps(acc_index: pd.DataFrame) -> tuple:
    """
    Pré-calcule les maps batch_same_hour, batch_same_day, batch_density.
    Retourne (hour_map, day_map, density_map) dict {aid: value}.
    """
    hour_map: dict    = {}
    day_map: dict     = {}
    density_map: dict = {}
    if acc_index.empty or AccountCols.CREATED_AT not in acc_index.columns:
        return hour_map, day_map, density_map
    try:
        created    = acc_index[AccountCols.CREATED_AT].dropna()
        ts         = pd.to_datetime(created, utc=True, errors="coerce").dropna()
        hour_str   = ts.dt.strftime("%Y-%m-%dT%H")
        day_str    = ts.dt.strftime("%Y-%m-%d")
        hour_counts = hour_str.value_counts()
        day_counts  = day_str.value_counts()
        n_total     = len(ts)
        for aid, h_str in hour_str.items():
            c = hour_counts.get(h_str, 1)
            hour_map[str(aid)]    = int(c >= 2)
            density_map[str(aid)] = round(c / max(n_total, 1), 4)
        for aid, d_str in day_str.items():
            day_map[str(aid)] = int(day_counts.get(d_str, 1) >= 2)
    except Exception as exc:
        logger.debug("[structural] batch_v11 erreur : %s", exc)
    return hour_map, day_map, density_map


# F3 — Cohérence username / bio / id
def _profile_coherence_features(row: pd.Series, aid: str) -> dict:
    """V1.1 F3 : ratios et flags de cohérence inter-champs du profil."""
    uname = str(row.get(AccountCols.SCREEN_NAME, "")) if pd.notna(
        row.get(AccountCols.SCREEN_NAME)) else ""
    bio   = str(row.get(AccountCols.BIO, "")) if pd.notna(row.get(AccountCols.BIO)) else ""
    id_s  = str(aid)
    # Ratio longueur username / bio
    if len(bio) > 0:
        u_b_ratio = round(len(uname) / len(bio), 4)
    else:
        u_b_ratio = 0.0
    # Username se termine par chiffre
    num_suffix = int(len(uname) > 0 and uname[-1].isdigit())
    # Ratio longueur id / username
    if len(uname) > 0:
        id_u_ratio = round(len(id_s) / len(uname), 4)
    else:
        id_u_ratio = 0.0
    # Complétude du profil : bio + location + name
    has_bio  = int(len(bio.strip()) > 5)
    has_loc  = int(len(str(row.get(AccountCols.LOCATION, "") or "").strip()) > 0)
    has_name = int(len(str(row.get(AccountCols.NAME, "") or "").strip()) > 0)
    completeness = round((has_bio + has_loc + has_name) / 3.0, 4)
    return {
        "str_username_bio_len_ratio":  u_b_ratio,
        "str_username_numeric_suffix": num_suffix,
        "str_id_username_len_ratio":   id_u_ratio,
        "str_profile_completeness":    completeness,
    }


# F4 — Profil template-like
def _template_features(row: pd.Series) -> dict:
    """V1.1 F4 : signaux de profil généré / template."""
    uname = str(row.get(AccountCols.SCREEN_NAME, "")) if pd.notna(
        row.get(AccountCols.SCREEN_NAME)) else ""
    bio   = str(row.get(AccountCols.BIO, "")) if pd.notna(row.get(AccountCols.BIO)) else ""
    # Bio très courte ou vide = template
    bio_is_template = int(len(bio.strip()) <= 10)
    # Username underscores
    n_underscores   = uname.count("_")
    # Longueur username
    uname_len       = len(uname)
    # Ratio chiffres dans username
    n_dig = sum(c.isdigit() for c in uname)
    dig_ratio = round(n_dig / max(len(uname), 1), 4)
    # Score composite (0-4)
    score = (
        bio_is_template
        + int(n_underscores >= 2)
        + int(uname_len > 12)
        + int(dig_ratio >= 0.25)
    )
    return {
        "str_bio_is_template":        bio_is_template,
        "str_username_underscores":   n_underscores,
        "str_username_len":           uname_len,
        "str_username_digits_ratio":  dig_ratio,
        "str_profile_template_score": score,
    }


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def extract_structural_features(
    accounts_df: Optional[pd.DataFrame] = None,
    posts_df:    Optional[pd.DataFrame] = None,
    cfg:         Optional[dict]         = None,
) -> pd.DataFrame:
    """
    Extrait les features structurelles pour chaque compte.

    Le module est opportuniste :
        - Si aucune colonne utile n'est trouvee -> retourne DataFrame vide propre
        - Si seulement certaines colonnes existent -> calcule ce qui est possible
        - Ne plante jamais

    V1.1 (features experimentales) :
        Activer via cfg={"structural": {"source_v11_enabled": True, ...}}
        Desactivees par defaut (RULES.md §2).

    Args:
        accounts_df : DataFrame de comptes (colonnes canoniques)
        posts_df    : DataFrame de posts
        cfg         : Dict de configuration (issu de features.yaml).  None = V1 pure.

    Returns:
        DataFrame avec account_id + features structurelles.
        Une ligne par compte. DataFrame vide si rien n'est dispo.
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

    # ── V1.1 flags ──────────────────────────────────────────────────────────
    str_cfg = (cfg or {}).get("structural", {})
    use_source_v11   = bool(str_cfg.get("source_v11_enabled",   False))
    use_batch_v11    = bool(str_cfg.get("batch_v11_enabled",    False))
    use_profile_v11  = bool(str_cfg.get("profile_v11_enabled",  False))
    use_template_v11 = bool(str_cfg.get("template_v11_enabled", False))

    # Pre-calcul V1.1 batch heure/jour (global, fait une seule fois)
    hour_map: dict  = {}
    day_map: dict   = {}
    density_map: dict = {}
    if use_batch_v11 and can_compute_batch:
        hour_map, day_map, density_map = _batch_features_v11_build_maps(acc_index)

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

        # Source features (V1)
        if aid in posts_grouped:
            feat.update(_source_features(posts_grouped[aid]))
        else:
            for col in ["str_source_n_unique", "str_source_is_single",
                        "str_source_has_api", "str_source_entropy"]:
                feat[col] = np.nan

        # Batch (V1)
        feat["str_batch_score"]        = global_batch_score
        feat["str_created_same_minute"] = float(batch_minute_map.get(aid, np.nan))

        # ── V1.1 features (derriere flags) ──────────────────────────────────
        # F1 : SOURCE V1.1
        if use_source_v11:
            try:
                if aid in posts_grouped:
                    feat.update(_source_features_v11(posts_grouped[aid]))
                else:
                    feat.update({k: np.nan for k in STRUCTURAL_COLS_V11["source_v11"]})
            except Exception as exc:
                logger.debug("[structural] source_v11 error '%s': %s", aid, exc)
                feat.update({k: np.nan for k in STRUCTURAL_COLS_V11["source_v11"]})

        # F2 : BATCH V1.1
        if use_batch_v11:
            feat["str_batch_same_hour"] = float(hour_map.get(aid, np.nan))
            feat["str_batch_same_day"]  = float(day_map.get(aid, np.nan))
            feat["str_batch_density"]   = float(density_map.get(aid, np.nan))

        # F3 : PROFILE COHERENCE V1.1
        if use_profile_v11 and can_compute_profile and aid in acc_index.index:
            try:
                acc_row = acc_index.loc[aid]
                if isinstance(acc_row, pd.DataFrame):
                    acc_row = acc_row.iloc[0]
                feat.update(_profile_coherence_features(acc_row, aid))
            except Exception as exc:
                logger.debug("[structural] profile_v11 error '%s': %s", aid, exc)
                feat.update({k: np.nan for k in STRUCTURAL_COLS_V11["profile_v11"]})
        elif use_profile_v11:
            feat.update({k: np.nan for k in STRUCTURAL_COLS_V11["profile_v11"]})

        # F4 : TEMPLATE V1.1
        if use_template_v11 and can_compute_profile and aid in acc_index.index:
            try:
                acc_row = acc_index.loc[aid]
                if isinstance(acc_row, pd.DataFrame):
                    acc_row = acc_row.iloc[0]
                feat.update(_template_features(acc_row))
            except Exception as exc:
                logger.debug("[structural] template_v11 error '%s': %s", aid, exc)
                feat.update({k: np.nan for k in STRUCTURAL_COLS_V11["template_v11"]})
        elif use_template_v11:
            feat.update({k: np.nan for k in STRUCTURAL_COLS_V11["template_v11"]})

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
