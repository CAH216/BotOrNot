# -*- coding: utf-8 -*-
"""
src/preprocessing/clean_text.py
--------------------------------
Nettoyage du texte brut des posts.

Principe :
    - text_raw  : texte original intact (jamais écrasé)
    - text_clean : version normalisée, prête pour les features NLP

Fonctions publiques :
    clean_text(text)           → str nettoyé
    clean_posts_df(df)         → DataFrame avec text_raw + text_clean
    extract_text_entities(text) → dict (urls, mentions, hashtags)
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Dict, List, Optional

import pandas as pd

from src.data.schema import PostCols

# ---------------------------------------------------------------------------
# Patterns regex compilés une seule fois
# ---------------------------------------------------------------------------

_RE_URL        = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION    = re.compile(r"@[\w_]+")
_RE_HASHTAG    = re.compile(r"#[\w_]+")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_HTML_TAG   = re.compile(r"<[^>]+>")
_RE_NEWLINE    = re.compile(r"[\r\n\t]+")
# Caractères de contrôle invisibles (sauf espace standard)
_RE_CTRL       = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Extraction des entités
# ---------------------------------------------------------------------------

def extract_text_entities(text: str) -> Dict[str, List[str]]:
    """
    Extrait les URLs, mentions et hashtags d'un texte brut.

    Args:
        text : texte brut

    Returns:
        {
            "urls"     : [...],
            "mentions" : [...],
            "hashtags" : [...],
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {"urls": [], "mentions": [], "hashtags": []}

    return {
        "urls"     : _RE_URL.findall(text),
        "mentions" : _RE_MENTION.findall(text),
        "hashtags" : _RE_HASHTAG.findall(text),
    }


# ---------------------------------------------------------------------------
# Nettoyage unitaire
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    remove_urls:     bool = True,
    remove_mentions: bool = False,
    remove_hashtags: bool = False,
    normalize_unicode: bool = True,
    lowercase:       bool = False,
) -> str:
    """
    Nettoie un texte brut.

    Args:
        text              : texte brut (str ou NaN-like)
        remove_urls       : remplacer les URLs par __URL__
        remove_mentions   : remplacer les @mentions par __MENTION__
        remove_hashtags   : remplacer les #hashtags par __HASHTAG__
        normalize_unicode : normaliser les caractères Unicode (NFKC)
        lowercase         : mettre en minuscules

    Returns:
        Texte nettoyé (str). Retourne "" si l'entrée est invalide.
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    # 1. Décodage HTML (&amp; → &, &lt; → <, etc.)
    text = html.unescape(text)

    # 2. Suppression des balises HTML résiduelles
    text = _RE_HTML_TAG.sub(" ", text)

    # 3. Suppression des caractères de contrôle
    text = _RE_CTRL.sub("", text)

    # 4. Normalisation des retours à la ligne
    text = _RE_NEWLINE.sub(" ", text)

    # 5. Normalisation Unicode (NFKC : ﬁ → fi, ½ → 1/2, etc.)
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    # 6. Gestion des URLs
    if remove_urls:
        text = _RE_URL.sub(" __URL__ ", text)

    # 7. Gestion des mentions
    if remove_mentions:
        text = _RE_MENTION.sub(" __MENTION__ ", text)

    # 8. Gestion des hashtags
    if remove_hashtags:
        text = _RE_HASHTAG.sub(" __HASHTAG__ ", text)

    # 9. Correction d'encodage Latin-1/Windows-1252 échappés
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass   # texte déjà en UTF-8 valide

    # 10. Normalisation des espaces
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # 11. Minuscules (si demandé)
    if lowercase:
        text = text.lower()

    return text


# ---------------------------------------------------------------------------
# Nettoyage sur un DataFrame entier
# ---------------------------------------------------------------------------

def clean_posts_df(
    df: pd.DataFrame,
    text_col:        Optional[str] = None,
    remove_urls:     bool = True,
    remove_mentions: bool = False,
    remove_hashtags: bool = False,
    normalize_unicode: bool = True,
    lowercase:       bool = False,
    extract_entities: bool = True,
) -> pd.DataFrame:
    """
    Applique le nettoyage de texte sur un DataFrame de posts.

    Comportement :
        - Crée `text_raw` (copie du texte original, inchangée)
        - Crée `text_clean` (version nettoyée)
        - Si extract_entities=True, crée `entities` (dict urls/mentions/hashtags)
          et des colonnes séparées `urls`, `mentions`, `hashtags`

    Args:
        df               : DataFrame de posts (doit contenir la colonne text)
        text_col         : nom de la colonne texte (défaut : PostCols.TEXT)
        remove_urls      : remplacer URLs par __URL__
        remove_mentions  : remplacer @mentions par __MENTION__
        remove_hashtags  : remplacer #hashtags par __HASHTAG__
        normalize_unicode: normaliser Unicode
        lowercase        : mettre en minuscules
        extract_entities : extraire URLs/mentions/hashtags dans des colonnes séparées

    Returns:
        DataFrame avec colonnes supplémentaires : text_raw, text_clean,
        et optionnellement urls, mentions, hashtags.
    """
    df = df.copy()
    col = text_col or PostCols.TEXT

    if col not in df.columns:
        # Pas de colonne texte → on retourne le df tel quel
        return df

    # 1. Préserver le texte brut
    if PostCols.TEXT_CLEAN.replace("_clean", "_raw") not in df.columns:
        df["text_raw"] = df[col].astype(str)

    # 2. Extraction des entités AVANT nettoyage
    if extract_entities:
        entities = df[col].fillna("").astype(str).apply(extract_text_entities)
        df["urls"]     = entities.apply(lambda e: e["urls"])
        df["mentions"] = entities.apply(lambda e: e["mentions"])
        df["hashtags"] = entities.apply(lambda e: e["hashtags"])

    # 3. Nettoyage
    df[PostCols.TEXT_CLEAN] = df[col].fillna("").astype(str).apply(
        lambda t: clean_text(
            t,
            remove_urls      = remove_urls,
            remove_mentions  = remove_mentions,
            remove_hashtags  = remove_hashtags,
            normalize_unicode= normalize_unicode,
            lowercase        = lowercase,
        )
    )

    return df
