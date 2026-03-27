# -*- coding: utf-8 -*-
"""
src/features/text_basic.py
---------------------------
Extracteur de features textuelles légères — sans GPU, sans gros modèle.

Ce module exploite la répétition, la diversité lexicale et la structure
superficielle du texte. Les signaux de bot les plus forts viennent souvent
d'une **homogénéité anormale** des posts : mêmes formulations, mêmes
longueurs, copier-coller quasi-identiques.

Features extraites (toutes préfixées "txt_") :
    Surface        — len_mean, len_std, uppercase_ratio, punct_mean
    Diversité      — ttr (type-token ratio), vocab_size
    Entités        — hashtags_mean, mentions_mean, urls_mean
    Répétition     — ngram2_repeat_rate, ngram3_repeat_rate
    Similarité     — sim_mean, sim_max, quasi_dup_rate

Usage :
    from src.features.text_basic import extract_text_features

    feat_df = extract_text_features(posts_df)
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.data.schema import AccountCols, PostCols

logger = logging.getLogger(__name__)

_PREFIX = "txt_"

# Seuil de similarité Jaccard pour quasi-duplication
_QUASI_DUP_THRESHOLD = 0.85

# Taille max d'échantillon pour le calcul de similarité intra-compte
# (O(n²) → pour de grands comptes on échantillonne)
_SIM_MAX_SAMPLE    = 50
_SIM_SAMPLE_UNIQUE = 30   # n-grams de chars uniques à comparer


# ---------------------------------------------------------------------------
# Utilitaires de tokenisation légère
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Tokenisation simple : lowercase, split sur espace, sans ponctuation."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+|@\w+|#\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extrait des n-grammes de caractères (pour similarité rapide)."""
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _ngrams_tokens(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extrait des n-grammes de tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Similarité de Jaccard entre deux ensembles."""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Calcul des features pour un groupe de posts d'un compte
# ---------------------------------------------------------------------------

def _compute_group_features(texts: List[str]) -> dict:
    """
    Calcule toutes les features textuelles pour un ensemble de posts.

    Args:
        texts : liste de textes bruts (non vides)

    Returns:
        dict de features préfixées
    """
    n = len(texts)
    f: dict = {f"{_PREFIX}n_posts": n}

    if n == 0:
        return _null_features()

    # --- Longueurs ---
    lengths = [len(t) for t in texts]
    f[f"{_PREFIX}len_mean"] = round(float(np.mean(lengths)), 2)
    f[f"{_PREFIX}len_std"]  = round(float(np.std(lengths)), 2) if n > 1 else 0.0
    f[f"{_PREFIX}len_min"]  = int(min(lengths))
    f[f"{_PREFIX}len_max"]  = int(max(lengths))

    # --- Majuscules ---
    def _uppercase_ratio(t: str) -> float:
        letters = [c for c in t if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    up_ratios = [_uppercase_ratio(t) for t in texts]
    f[f"{_PREFIX}uppercase_ratio"] = round(float(np.mean(up_ratios)), 4)

    # --- Ponctuation moyenne (nombre de signes de ponctuation par post) ---
    punct_set = set(string.punctuation)
    punct_counts = [sum(1 for c in t if c in punct_set) for t in texts]
    f[f"{_PREFIX}punct_mean"] = round(float(np.mean(punct_counts)), 4)

    # --- Type-Token Ratio (TTR) global ---
    all_tokens = []
    per_post_tokens = []
    for t in texts:
        toks = _tokenize(t)
        all_tokens.extend(toks)
        per_post_tokens.append(toks)

    n_tokens = len(all_tokens)
    n_types  = len(set(all_tokens))
    f[f"{_PREFIX}vocab_size"] = n_types
    f[f"{_PREFIX}ttr"]        = round(float(n_types / max(n_tokens, 1)), 4)

    # --- Hashtags, mentions, URLs ---
    _re_ht  = re.compile(r"#\w+")
    _re_mt  = re.compile(r"@\w+")
    _re_url = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

    f[f"{_PREFIX}hashtags_mean"] = round(float(np.mean([len(_re_ht.findall(t)) for t in texts])), 4)
    f[f"{_PREFIX}mentions_mean"] = round(float(np.mean([len(_re_mt.findall(t)) for t in texts])), 4)
    f[f"{_PREFIX}urls_mean"]     = round(float(np.mean([len(_re_url.findall(t)) for t in texts])), 4)

    # --- Répétition de n-grammes de tokens ---
    # Taux de n-grammes qui apparaissent plus d'une fois dans le corpus du compte
    def _ngram_repeat_rate(posts_tokens: List[List[str]], n: int) -> float:
        all_ng = []
        for toks in posts_tokens:
            all_ng.extend(_ngrams_tokens(toks, n))
        if not all_ng:
            return 0.0
        counts = Counter(all_ng)
        repeated = sum(v for v in counts.values() if v > 1)
        return round(float(repeated / len(all_ng)), 4)

    f[f"{_PREFIX}ngram2_repeat_rate"] = _ngram_repeat_rate(per_post_tokens, 2)
    f[f"{_PREFIX}ngram3_repeat_rate"] = _ngram_repeat_rate(per_post_tokens, 3)

    # --- Similarité intra-compte (Jaccard sur char 3-grammes) ---
    # Échantillonnage pour les grands comptes (O(n²))
    sample_texts = texts if n <= _SIM_MAX_SAMPLE else _reservoir_sample(texts, _SIM_MAX_SAMPLE)
    ng_sets = [_char_ngrams(t, 3) for t in sample_texts]
    ns = len(ng_sets)

    sims: List[float] = []
    if ns >= 2:
        for i in range(ns):
            for j in range(i + 1, ns):
                sims.append(_jaccard(ng_sets[i], ng_sets[j]))

    if sims:
        f[f"{_PREFIX}sim_mean"] = round(float(np.mean(sims)), 4)
        f[f"{_PREFIX}sim_max"]  = round(float(np.max(sims)), 4)
        # Quasi-duplication : fraction de paires dépassant le seuil
        f[f"{_PREFIX}quasi_dup_rate"] = round(
            float(sum(1 for s in sims if s >= _QUASI_DUP_THRESHOLD) / len(sims)), 4
        )
    else:
        f[f"{_PREFIX}sim_mean"]       = 0.0
        f[f"{_PREFIX}sim_max"]        = 0.0
        f[f"{_PREFIX}quasi_dup_rate"] = 0.0

    return f


def _null_features() -> dict:
    """Retourne un dict de toutes les features à NaN."""
    return {
        f"{_PREFIX}n_posts":            0,
        f"{_PREFIX}len_mean":           np.nan,
        f"{_PREFIX}len_std":            np.nan,
        f"{_PREFIX}len_min":            np.nan,
        f"{_PREFIX}len_max":            np.nan,
        f"{_PREFIX}uppercase_ratio":    np.nan,
        f"{_PREFIX}punct_mean":         np.nan,
        f"{_PREFIX}vocab_size":         0,
        f"{_PREFIX}ttr":                np.nan,
        f"{_PREFIX}hashtags_mean":      np.nan,
        f"{_PREFIX}mentions_mean":      np.nan,
        f"{_PREFIX}urls_mean":          np.nan,
        f"{_PREFIX}ngram2_repeat_rate": np.nan,
        f"{_PREFIX}ngram3_repeat_rate": np.nan,
        f"{_PREFIX}sim_mean":           np.nan,
        f"{_PREFIX}sim_max":            np.nan,
        f"{_PREFIX}quasi_dup_rate":     np.nan,
    }


def _reservoir_sample(lst: list, k: int) -> list:
    """Reservoir sampling pour échantillon uniforme sans copie complète."""
    import random
    reservoir = lst[:k]
    for i in range(k, len(lst)):
        j = random.randint(0, i)
        if j < k:
            reservoir[j] = lst[i]
    return reservoir


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def extract_text_features(
    posts_df:    pd.DataFrame,
    accounts_df: Optional[pd.DataFrame] = None,
    text_col:    Optional[str] = None,
    min_posts:   int = 1,
) -> pd.DataFrame:
    """
    Extrait les features textuelles légères pour chaque compte.

    Args:
        posts_df    : DataFrame de posts (doit contenir account_id + texte)
        accounts_df : optionnel — pour créer des lignes NaN pour comptes sans posts
        text_col    : colonne texte à utiliser (défaut : text_clean si dispo, sinon text)
        min_posts   : nb minimum de posts pour calculer les features (sinon NaN)

    Returns:
        DataFrame avec account_id + features textuelles.
        Une ligne par compte.
    """
    id_col = AccountCols.ID

    if posts_df is None or posts_df.empty:
        logger.warning("posts_df vide — features textuelles nulles")
        return pd.DataFrame(columns=[id_col] + list(_null_features().keys()))

    if id_col not in posts_df.columns:
        logger.warning("Colonne account_id absente dans posts_df")
        return pd.DataFrame(columns=[id_col] + list(_null_features().keys()))

    # Choisir la colonne texte
    if text_col is None:
        if PostCols.TEXT_CLEAN in posts_df.columns:
            text_col = PostCols.TEXT_CLEAN
        elif PostCols.TEXT in posts_df.columns:
            text_col = PostCols.TEXT
        else:
            logger.warning("Aucune colonne texte trouvée dans posts_df")
            return pd.DataFrame(columns=[id_col] + list(_null_features().keys()))

    rows = []
    for account_id, group in posts_df.groupby(id_col, sort=False):
        feat = {id_col: account_id}

        texts = group[text_col].fillna("").astype(str).tolist()
        # Filtrer les textes vides
        texts = [t for t in texts if t.strip()]

        if len(texts) < min_posts:
            feat.update(_null_features())
            feat[f"{_PREFIX}n_posts"] = len(group)
        else:
            try:
                feat.update(_compute_group_features(texts))
            except Exception as e:
                logger.warning("Erreur features texte compte '%s': %s", account_id, e)
                feat.update(_null_features())

        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=[id_col] + list(_null_features().keys()))

    result_df = pd.DataFrame(rows).reset_index(drop=True)

    # Comptes sans posts (si accounts_df fourni)
    if accounts_df is not None and id_col in accounts_df.columns:
        known = set(result_df[id_col])
        missing_ids = set(accounts_df[id_col]) - known
        if missing_ids:
            null_rows = []
            for aid in missing_ids:
                row = {id_col: aid}
                row.update(_null_features())
                null_rows.append(row)
            result_df = pd.concat(
                [result_df, pd.DataFrame(null_rows)],
                ignore_index=True
            )

    # Typage numérique
    num_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[num_cols] = result_df[num_cols].astype(np.float32)

    logger.info(
        "Features texte (léger) : %d comptes × %d features",
        len(result_df), len(result_df.columns) - 1,
    )
    return result_df
