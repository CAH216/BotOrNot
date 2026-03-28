# -*- coding: utf-8 -*-
"""
src/features/tabular.py
------------------------
Extracteur de features tabulaires V1.1 — couche principale du pipeline.

Ce module extrait les features numériques et catégorielles à partir de :
    - accounts_df : profil du compte
    - posts_df    : agrégats sur les publications

Ce bloc est TOUJOURS actif. Il doit produire quelque chose même si
la plupart des colonnes sont absentes (retourne 0 / NaN pour les
colonnes non disponibles, ne plante jamais).

Features extraites :
    Profil compte  — âge, total_posts, fréquence, username, bio, followers
    Agrégats posts — longueur, hashtags, mentions, liens, réponses, reposts
    Ratios dérivés — reply_ratio, url_ratio, mention_ratio, digit_ratio

Usage :
    from src.features.tabular import extract_tabular_features

    feat_df = extract_tabular_features(accounts_df, posts_df)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import AccountCols, PostCols

logger = logging.getLogger(__name__)

# Seuil en jours pour considérer un compte comme "nouveau"
_NEW_ACCOUNT_THRESHOLD_DAYS = 30

# Préfixe des colonnes pour l'assembleur
_PREFIX = "tab_"


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _safe_div(a, b, default: float = 0.0) -> float:
    """Division sécurisée : retourne default si b == 0 ou NaN."""
    try:
        if pd.isna(b) or b == 0:
            return default
        return float(a) / float(b)
    except Exception:
        return default


def _count_digits(s: str) -> int:
    """Compte les chiffres dans une chaîne."""
    if not isinstance(s, str):
        return 0
    return sum(c.isdigit() for c in s)


def _count_hashtags(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"#\w+", text))


def _count_mentions(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"@\w+", text))


def _count_urls(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"https?://\S+|www\.\S+", text, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Features extraites de accounts_df
# ---------------------------------------------------------------------------

def _account_features(row: pd.Series, now: pd.Timestamp) -> dict:
    """
    Calcule les features de profil pour un compte.

    Args:
        row : ligne d'accounts_df (colonnes canoniques)
        now : timestamp de référence pour le calcul d'âge

    Returns:
        dict de features préfixées
    """
    f: dict = {}

    # --- Âge du compte (jours depuis création) ---
    created = row.get(AccountCols.CREATED_AT)
    if pd.notna(created):
        try:
            created_ts = pd.Timestamp(created)
            if created_ts.tzinfo is None:
                created_ts = created_ts.tz_localize("UTC")
            if now.tzinfo is None:
                now = now.tz_localize("UTC")
            age_days = max(float((now - created_ts).days), 0.0)
        except Exception:
            age_days = np.nan
    else:
        age_days = np.nan
    f[f"{_PREFIX}account_age_days"] = age_days
    f[f"{_PREFIX}is_new_account"]   = (
        int(age_days < _NEW_ACCOUNT_THRESHOLD_DAYS)
        if not np.isnan(age_days) else np.nan
    )

    # --- Activité globale ---
    total_posts = float(row.get(AccountCols.TOTAL_POSTS, np.nan))
    f[f"{_PREFIX}total_posts"] = total_posts

    # Fréquence : posts par jour depuis création
    if not np.isnan(total_posts) and not np.isnan(age_days) and age_days > 0:
        f[f"{_PREFIX}post_frequency"] = round(_safe_div(total_posts, age_days), 4)
    else:
        f[f"{_PREFIX}post_frequency"] = np.nan

    # --- Followers / Following ---
    followers = float(row.get(AccountCols.FOLLOWERS, np.nan))
    following = float(row.get(AccountCols.FOLLOWING, np.nan))
    f[f"{_PREFIX}followers"]       = followers
    f[f"{_PREFIX}following"]       = following
    f[f"{_PREFIX}follower_ratio"]  = round(_safe_div(followers, following + 1), 4)
    f[f"{_PREFIX}follow_back_gap"] = round(
        float(followers - following)
        if (not np.isnan(followers) and not np.isnan(following)) else np.nan,
        2
    )

    # --- Bio ---
    bio = row.get(AccountCols.BIO, "")
    bio_str = str(bio) if pd.notna(bio) else ""
    f[f"{_PREFIX}bio_len"]       = len(bio_str)
    f[f"{_PREFIX}bio_is_empty"]  = int(bio_str.strip() == "")

    # --- Username ---
    username = row.get(AccountCols.SCREEN_NAME, "")
    uname = str(username) if pd.notna(username) else ""
    n_digits = _count_digits(uname)
    f[f"{_PREFIX}username_len"]         = len(uname)
    f[f"{_PREFIX}username_n_digits"]    = n_digits
    f[f"{_PREFIX}username_digit_ratio"] = round(
        _safe_div(n_digits, max(len(uname), 1)), 4
    )
    # Underscores dans username (signal de bot-name generation)
    f[f"{_PREFIX}username_n_underscores"] = uname.count("_")

    # --- Compte vérifié ---
    verified = row.get(AccountCols.VERIFIED, np.nan)
    f[f"{_PREFIX}is_verified"] = (
        int(bool(verified)) if pd.notna(verified) else np.nan
    )

    # --- Image de profil par défaut ---
    default_img = row.get(AccountCols.PROFILE_IMAGE, np.nan)
    f[f"{_PREFIX}has_default_image"] = (
        int(bool(default_img)) if pd.notna(default_img) else np.nan
    )

    # ── V1.1 — Log transforms ─────────────────────────────────────────────
    # Stabilisent la distribution pour les modèles de type arbre et LR.
    f[f"{_PREFIX}followers_log"]  = float(np.log1p(max(followers, 0))) if not np.isnan(followers) else np.nan
    f[f"{_PREFIX}following_log"]  = float(np.log1p(max(following, 0))) if not np.isnan(following) else np.nan
    f[f"{_PREFIX}total_posts_log"]= float(np.log1p(max(total_posts, 0))) if not np.isnan(total_posts) else np.nan
    if not np.isnan(age_days):
        f[f"{_PREFIX}age_days_log"] = float(np.log1p(max(age_days, 0)))
    else:
        f[f"{_PREFIX}age_days_log"] = np.nan

    # ── V1.1 — Flags extrêmes ────────────────────────────────────────────
    # Indicateurs booléens pour détecter des valeurs anormalement hautes.
    if not np.isnan(followers):
        f[f"{_PREFIX}followers_extreme"] = int(followers > 100_000)
        f[f"{_PREFIX}followers_zero"]    = int(followers == 0)
    else:
        f[f"{_PREFIX}followers_extreme"] = np.nan
        f[f"{_PREFIX}followers_zero"]    = np.nan

    if not np.isnan(following):
        f[f"{_PREFIX}following_extreme"] = int(following > 5_000)
    else:
        f[f"{_PREFIX}following_extreme"] = np.nan

    if not np.isnan(total_posts):
        f[f"{_PREFIX}statuses_extreme"] = int(total_posts > 100_000)
    else:
        f[f"{_PREFIX}statuses_extreme"] = np.nan

    # ── V1.1 — Productivité par follower ─────────────────────────────────
    # Bots : beaucoup de posts, peu de followers.
    if not (np.isnan(total_posts) or np.isnan(followers)):
        f[f"{_PREFIX}posts_per_follower"] = round(
            _safe_div(total_posts, followers + 1), 4
        )
        f[f"{_PREFIX}posts_per_follower_log"] = float(
            np.log1p(_safe_div(total_posts, followers + 1))
        )
    else:
        f[f"{_PREFIX}posts_per_follower"]     = np.nan
        f[f"{_PREFIX}posts_per_follower_log"] = np.nan

    return f


# ---------------------------------------------------------------------------
# Features agrégées de posts_df
# ---------------------------------------------------------------------------

def _posts_agg_features(group: pd.DataFrame) -> dict:
    """
    Calcule les features d'agrégat pour un groupe de posts d'un compte.

    Args:
        group : sous-DataFrame posts pour un compte

    Returns:
        dict de features préfixées
    """
    f: dict = {}
    n = len(group)
    f[f"{_PREFIX}n_posts_observed"] = n

    if n == 0:
        # Colonnes avec 0 pour comptes sans posts
        for col in [
            "post_len_mean", "post_len_std", "post_len_min", "post_len_max",
            "hashtags_mean", "mentions_mean", "urls_mean",
            "reply_count", "repost_count", "reply_ratio", "repost_ratio",
            "source_diversity",
        ]:
            f[f"{_PREFIX}{col}"] = 0.0
        return f

    # --- Longueurs de posts ---
    if PostCols.TEXT in group.columns:
        texts = group[PostCols.TEXT].fillna("").astype(str)
        lengths = texts.str.len()
        f[f"{_PREFIX}post_len_mean"] = round(float(lengths.mean()), 2)
        f[f"{_PREFIX}post_len_std"]  = round(float(lengths.std()), 2) if n > 1 else 0.0
        f[f"{_PREFIX}post_len_min"]  = int(lengths.min())
        f[f"{_PREFIX}post_len_max"]  = int(lengths.max())
    else:
        for col in ["post_len_mean", "post_len_std", "post_len_min", "post_len_max"]:
            f[f"{_PREFIX}{col}"] = np.nan

    # --- Hashtags, mentions, URLs ---
    # Cas 1 : colonnes pré-extraites par clean_posts_df
    if "hashtags" in group.columns:
        f[f"{_PREFIX}hashtags_mean"] = round(float(
            group["hashtags"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        ), 4)
    elif PostCols.TEXT in group.columns:
        texts = group[PostCols.TEXT].fillna("").astype(str)
        f[f"{_PREFIX}hashtags_mean"] = round(float(texts.apply(_count_hashtags).mean()), 4)
    else:
        f[f"{_PREFIX}hashtags_mean"] = np.nan

    if "mentions" in group.columns:
        f[f"{_PREFIX}mentions_mean"] = round(float(
            group["mentions"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        ), 4)
    elif PostCols.TEXT in group.columns:
        texts = group[PostCols.TEXT].fillna("").astype(str)
        f[f"{_PREFIX}mentions_mean"] = round(float(texts.apply(_count_mentions).mean()), 4)
    else:
        f[f"{_PREFIX}mentions_mean"] = np.nan

    if "urls" in group.columns:
        f[f"{_PREFIX}urls_mean"] = round(float(
            group["urls"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        ), 4)
    elif PostCols.TEXT in group.columns:
        texts = group[PostCols.TEXT].fillna("").astype(str)
        f[f"{_PREFIX}urls_mean"] = round(float(texts.apply(_count_urls).mean()), 4)
    else:
        f[f"{_PREFIX}urls_mean"] = np.nan

    # --- Réponses et reposts ---
    is_reply  = group[PostCols.IN_REPLY_TO].notna() if PostCols.IN_REPLY_TO in group.columns else pd.Series([False]*n)
    is_repost = group[PostCols.RETWEET_OF].notna()  if PostCols.RETWEET_OF  in group.columns else pd.Series([False]*n)

    reply_count  = int(is_reply.sum())
    repost_count = int(is_repost.sum())
    f[f"{_PREFIX}reply_count"]   = reply_count
    f[f"{_PREFIX}repost_count"]  = repost_count
    f[f"{_PREFIX}reply_ratio"]   = round(_safe_div(reply_count, n), 4)
    f[f"{_PREFIX}repost_ratio"]  = round(_safe_div(repost_count, n), 4)

    # --- Diversité de la source ---
    if PostCols.SOURCE in group.columns:
        f[f"{_PREFIX}source_diversity"] = int(group[PostCols.SOURCE].nunique())
    else:
        f[f"{_PREFIX}source_diversity"] = np.nan

    return f


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def extract_tabular_features(
    accounts_df: Optional[pd.DataFrame] = None,
    posts_df:    Optional[pd.DataFrame] = None,
    reference_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extrait les features tabulaires pour chaque compte.

    Ce module s'adapte à ce qui est disponible :
        - Si accounts_df seul    → features de profil uniquement
        - Si posts_df seul       → features d'agrégats uniquement
        - Si les deux           → features complètes

    Args:
        accounts_df    : DataFrame de comptes normalisé
        posts_df       : DataFrame de posts normalisé
        reference_date : date de référence pour le calcul d'âge (ISO8601)
                         défaut : maintenant

    Returns:
        DataFrame avec account_id + toutes les features tabulaires.
        Une ligne par compte, NaN pour les features non calculables.
    """
    id_col = AccountCols.ID

    if accounts_df is None and posts_df is None:
        logger.warning("Aucune donnée fournie à extract_tabular_features")
        return pd.DataFrame(columns=[id_col])

    # Date de référence pour calculer l'âge du compte
    if reference_date:
        now = pd.Timestamp(reference_date, tz="UTC")
    else:
        now = pd.Timestamp.utcnow()

    # ---- 1. Construire la liste des account_ids ----
    known_ids: set = set()
    if accounts_df is not None and not accounts_df.empty and id_col in accounts_df.columns:
        known_ids.update(accounts_df[id_col].dropna().astype(str))
    if posts_df is not None and not posts_df.empty and id_col in posts_df.columns:
        known_ids.update(posts_df[id_col].dropna().astype(str))

    if not known_ids:
        return pd.DataFrame(columns=[id_col])

    # ---- 2. Index des accounts ----
    if accounts_df is not None and not accounts_df.empty and id_col in accounts_df.columns:
        acc_index = accounts_df.set_index(id_col)
    else:
        acc_index = pd.DataFrame()

    # ---- 3. Index des posts ----
    if posts_df is not None and not posts_df.empty and id_col in posts_df.columns:
        posts_grouped = {k: v for k, v in posts_df.groupby(id_col)}
    else:
        posts_grouped = {}

    # ---- 4. Calcul par compte ----
    rows = []
    for aid in sorted(known_ids):
        row_feats: dict = {id_col: aid}

        # Features de profil
        if len(acc_index) > 0 and aid in acc_index.index:
            acc_row = acc_index.loc[aid]
            # Si plusieurs lignes (doublon résiduel) → prendre la première
            if isinstance(acc_row, pd.DataFrame):
                acc_row = acc_row.iloc[0]
            try:
                row_feats.update(_account_features(acc_row, now))
            except Exception as e:
                logger.warning("Erreur features profil '%s': %s", aid, e)
        else:
            # Pas de données de profil → NaN
            for col in [
                "account_age_days", "is_new_account", "total_posts", "post_frequency",
                "followers", "following", "follower_ratio", "follow_back_gap",
                "bio_len", "bio_is_empty",
                "username_len", "username_n_digits", "username_digit_ratio",
                "username_n_underscores", "is_verified", "has_default_image",
            ]:
                row_feats[f"{_PREFIX}{col}"] = np.nan

        # Features d'agrégats de posts
        if aid in posts_grouped:
            group = posts_grouped[aid]
            try:
                row_feats.update(_posts_agg_features(group))
            except Exception as e:
                logger.warning("Erreur features posts '%s': %s", aid, e)
        else:
            row_feats.update(_posts_agg_features(pd.DataFrame()))

        # --- Features dérivées cross-source ---
        # Cohérence total_posts déclaré vs observé
        declared  = row_feats.get(f"{_PREFIX}total_posts", np.nan)
        observed  = row_feats.get(f"{_PREFIX}n_posts_observed", 0)
        if not np.isnan(declared) and observed > 0:
            row_feats[f"{_PREFIX}posts_observed_ratio"] = round(
                _safe_div(observed, max(declared, 1)), 4
            )
        else:
            row_feats[f"{_PREFIX}posts_observed_ratio"] = np.nan

        rows.append(row_feats)

    result_df = pd.DataFrame(rows).reset_index(drop=True)

    # Colonnes numériques → float32 pour économie mémoire
    num_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[num_cols] = result_df[num_cols].astype(np.float32)

    logger.info(
        "Features tabulaires : %d comptes × %d features",
        len(result_df), len(result_df.columns) - 1
    )
    return result_df
