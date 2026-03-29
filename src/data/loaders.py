# -*- coding: utf-8 -*-
"""
src/data/loaders.py
-------------------
Chargement et normalisation des donnees brutes vers un DataBundle.

Formats supportes :
    - CSV (une table plate ou deux tables comptes/posts)
    - JSON  (orientations : records, split, columns)
    - JSONL (un objet JSON par ligne)

Differents modes de chargement :
    load_file(path)            — auto-detecte le format, tente de separer comptes / posts
    load_accounts(path)        — charge un fichier oriente "comptes"
    load_posts(path)           — charge un fichier oriente "posts"
    load_multi(accounts, posts) — charge deux fichiers separes et les assemble
    load_bundle(...)           — point d'entree principal (recommande)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.schema import (
    AccountCols,
    DataBundle,
    LabelCols,
    PostCols,
)
from src.preprocessing.normalize_columns import normalize_columns as _normalize_cols

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilitaires bas niveau
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    """Retourne 'csv', 'json' ou 'jsonl' selon extension + contenu."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        # Peek pour distinguer JSON array/object vs JSONL
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_char = f.read(1).strip()
        return "jsonl" if first_char not in ("{", "[") else "json"
    raise ValueError(
        f"Format non reconnu : '{suffix}'. "
        "Fichiers supportes : .csv, .json, .jsonl"
    )


def _read_raw(path: Path, fmt: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Lit le fichier brut et retourne un DataFrame sans normalisation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    if fmt == "csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)

    if fmt == "json":
        try:
            df = pd.read_json(path, orient="records")
        except Exception:
            try:
                df = pd.read_json(path)
            except Exception as e:
                raise ValueError(f"Impossible de lire le JSON ({path}): {e}") from e
        return df.head(nrows) if nrows is not None else df

    if fmt == "jsonl":
        records = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if nrows is not None and i >= nrows:
                    break
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Ligne %d ignoree (JSON invalide): %s", i + 1, e)
        return pd.DataFrame(records)

    raise ValueError(f"Format interne inconnu : {fmt}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Délègue au module preprocessing.normalize_columns.
    Wrapper interne pour conserver la signature attendue dans loaders.py.
    """
    df, report = _normalize_cols(df)
    if report.n_renamed > 0:
        logger.debug("Normalisation : %d colonnes renommees — %s",
                     report.n_renamed, report.renamed)
    return df


def _parse_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convertit une colonne en datetime si elle existe."""
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        except Exception as e:
            logger.warning("Impossible de parser les dates de '%s': %s", col, e)
    return df


# ---------------------------------------------------------------------------
# Detection du type de table (comptes vs posts)
# ---------------------------------------------------------------------------

def _classify_table(df: pd.DataFrame) -> str:
    """
    Retourne 'accounts', 'posts' ou 'mixed' selon les colonnes presentes.
    Logique heuristique basee sur les colonnes canoniques.
    """
    cols = set(df.columns)

    account_signals = {AccountCols.ID, AccountCols.SCREEN_NAME,
                       AccountCols.BIO, AccountCols.FOLLOWERS,
                       AccountCols.FOLLOWING, AccountCols.TOTAL_POSTS}

    post_signals    = {PostCols.ID, PostCols.TEXT, PostCols.SOURCE,
                       PostCols.IN_REPLY_TO, PostCols.RETWEET_OF}

    n_account = len(cols & account_signals)
    n_post    = len(cols & post_signals)

    if n_account > n_post:
        return "accounts"
    if n_post > n_account:
        return "posts"
    # En cas d'ambiguite, si text est present c'est plutot posts
    if PostCols.TEXT in cols:
        return "posts"
    return "mixed"


# ---------------------------------------------------------------------------
# Separation comptes / posts depuis une table plate
# ---------------------------------------------------------------------------

def _split_flat_table(df: pd.DataFrame) -> tuple:
    """
    A partir d'une table "plate" (une ligne = un post avec info compte),
    tente d'extraire :
        - accounts_df : info unique par compte
        - posts_df    : une ligne par post
    """
    account_cols = [c for c in df.columns if c in {
        AccountCols.ID, AccountCols.SCREEN_NAME, AccountCols.BIO,
        AccountCols.FOLLOWERS, AccountCols.FOLLOWING,
        AccountCols.TOTAL_POSTS, AccountCols.VERIFIED,
        AccountCols.PROFILE_IMAGE, AccountCols.LOCATION, AccountCols.LANG,
    }]

    post_cols = [c for c in df.columns if c in {
        PostCols.ID, PostCols.TEXT, PostCols.TEXT_CLEAN,
        PostCols.CREATED_AT, PostCols.LANG, PostCols.SOURCE,
        PostCols.IN_REPLY_TO, PostCols.RETWEET_OF,
        PostCols.HASHTAGS, PostCols.MENTIONS, PostCols.URLS,
        AccountCols.ID,   # FK
    }]

    # accounts_df : deduplication par account_id
    if AccountCols.ID in account_cols:
        accounts_df = df[account_cols].drop_duplicates(
            subset=[AccountCols.ID]
        ).reset_index(drop=True)
    else:
        accounts_df = df[account_cols].drop_duplicates().reset_index(drop=True)

    # posts_df : toutes les lignes (chaque ligne = 1 post)
    posts_df = df[post_cols].reset_index(drop=True) if post_cols else None

    return accounts_df, posts_df


# ---------------------------------------------------------------------------
# Extraction des labels
# ---------------------------------------------------------------------------

def _extract_labels(df: pd.DataFrame) -> tuple:
    """
    Si une colonne label existe dans accounts_df ou posts_df,
    l'extraire dans un labels_df separe.
    Retourne (df_sans_label, labels_df | None).
    """
    if LabelCols.LABEL not in df.columns:
        return df, None

    key_col = AccountCols.ID if AccountCols.ID in df.columns else None

    if key_col:
        labels_df = df[[key_col, LabelCols.LABEL]].drop_duplicates(
            subset=[key_col]
        ).reset_index(drop=True)
        # Normalise le label : 1/0, True/False, "bot"/"human" → 1/0
        labels_df[LabelCols.LABEL] = _normalize_label(labels_df[LabelCols.LABEL])
        df = df.drop(columns=[LabelCols.LABEL])
    else:
        labels_df = df[[LabelCols.LABEL]].copy()
        labels_df[LabelCols.LABEL] = _normalize_label(labels_df[LabelCols.LABEL])
        df = df.drop(columns=[LabelCols.LABEL])

    return df, labels_df


def _normalize_label(series: pd.Series) -> pd.Series:
    """Convertit les labels textuels en 0/1."""
    mapping = {
        "bot": 1, "human": 0, "humain": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "1": 1, "0": 0,
    }
    if series.dtype == object:
        return series.str.lower().map(mapping).fillna(series).astype(float)
    return series.astype(float)


# ---------------------------------------------------------------------------
# Points d'entree publics
# ---------------------------------------------------------------------------

def load_file(
    path: str | Path,
    nrows: Optional[int] = None,
) -> DataBundle:
    """
    Charge un fichier unique (CSV/JSON/JSONL) et produit un DataBundle.

    Strategies d'interpretation (dans l'ordre) :
        1. Si la table semble etre uniquement des comptes → accounts_df
        2. Si la table semble etre uniquement des posts   → posts_df
        3. Sinon (table plate mixte)                      → split automatique

    Args:
        path  : chemin vers le fichier brut
        nrows : (optionnel) nombre de lignes a charger

    Returns:
        DataBundle normalise
    """
    path = Path(path)
    fmt  = _detect_format(path)
    logger.info("Chargement de %s (format=%s)", path.name, fmt)

    raw = _read_raw(path, fmt, nrows=nrows)
    df  = _normalize_columns(raw)
    df  = _parse_dates(df, AccountCols.CREATED_AT)
    df  = _parse_dates(df, PostCols.CREATED_AT)

    table_type = _classify_table(df)
    logger.info("Type de table detecte : %s", table_type)

    labels_df    = None
    edges_df     = None
    accounts_df  = None
    posts_df     = None

    if table_type == "accounts":
        df, labels_df = _extract_labels(df)
        accounts_df = df

    elif table_type == "posts":
        df, labels_df = _extract_labels(df)
        posts_df = df

    else:  # mixed / flat
        df, labels_df = _extract_labels(df)
        accounts_df, posts_df = _split_flat_table(df)

    bundle = DataBundle(
        accounts_df  = accounts_df,
        posts_df     = posts_df,
        edges_df     = edges_df,
        labels_df    = labels_df,
        source_path  = str(path.resolve()),
        source_format = fmt,
        n_accounts   = len(accounts_df) if accounts_df is not None else 0,
        n_posts      = len(posts_df)    if posts_df    is not None else 0,
    )
    logger.info("DataBundle cree : %d comptes, %d posts", bundle.n_accounts, bundle.n_posts)
    return bundle


def load_accounts(path: str | Path, nrows: Optional[int] = None) -> DataBundle:
    """
    Charge un fichier dont chaque ligne represente un compte.
    Force l'interpretation comme 'accounts'.
    """
    path = Path(path)
    fmt  = _detect_format(path)
    raw  = _read_raw(path, fmt, nrows=nrows)
    df   = _normalize_columns(raw)
    df   = _parse_dates(df, AccountCols.CREATED_AT)
    df, labels_df = _extract_labels(df)

    return DataBundle(
        accounts_df   = df,
        labels_df     = labels_df,
        source_path   = str(path.resolve()),
        source_format = fmt,
        n_accounts    = len(df),
    )


def load_posts(path: str | Path, nrows: Optional[int] = None) -> DataBundle:
    """
    Charge un fichier dont chaque ligne represente un post.
    Force l'interpretation comme 'posts'.
    """
    path = Path(path)
    fmt  = _detect_format(path)
    raw  = _read_raw(path, fmt, nrows=nrows)
    df   = _normalize_columns(raw)
    df   = _parse_dates(df, PostCols.CREATED_AT)
    df, labels_df = _extract_labels(df)

    return DataBundle(
        posts_df      = df,
        labels_df     = labels_df,
        source_path   = str(path.resolve()),
        source_format = fmt,
        n_posts       = len(df),
    )


def load_multi(
    accounts_path: str | Path,
    posts_path: Optional[str | Path] = None,
    edges_path:  Optional[str | Path] = None,
    labels_path: Optional[str | Path] = None,
    nrows: Optional[int] = None,
) -> DataBundle:
    """
    Charge plusieurs fichiers separes (comptes, posts, edges, labels).
    Mode recommande quand le dataset fournit des fichiers distincts.

    Args:
        accounts_path : fichier des comptes (obligatoire)
        posts_path    : fichier des posts (optionnel)
        edges_path    : fichier des relations (optionnel)
        labels_path   : fichier des labels (optionnel)
        nrows         : limite de lignes par fichier
    """
    # Comptes
    acc_bundle = load_accounts(accounts_path, nrows=nrows)
    accounts_df = acc_bundle.accounts_df
    labels_df   = acc_bundle.labels_df

    # Posts
    posts_df = None
    if posts_path:
        p = Path(posts_path)
        fmt = _detect_format(p)
        raw = _read_raw(p, fmt, nrows=nrows)
        df  = _normalize_columns(raw)
        df  = _parse_dates(df, PostCols.CREATED_AT)
        df, post_labels = _extract_labels(df)
        posts_df = df
        # Si pas de labels dans accounts, on prend ceux des posts
        if labels_df is None and post_labels is not None:
            labels_df = post_labels

    # Edges
    edges_df = None
    if edges_path:
        e = Path(edges_path)
        fmt = _detect_format(e)
        raw = _read_raw(e, fmt, nrows=nrows)
        edges_df = _normalize_columns(raw)

    # Labels separes (prioritaire sur les autres)
    if labels_path:
        l = Path(labels_path)
        fmt = _detect_format(l)
        raw = _read_raw(l, fmt, nrows=nrows)
        df  = _normalize_columns(raw)
        df[LabelCols.LABEL] = _normalize_label(df[LabelCols.LABEL])
        labels_df = df

    return DataBundle(
        accounts_df   = accounts_df,
        posts_df      = posts_df,
        edges_df      = edges_df,
        labels_df     = labels_df,
        source_path   = str(Path(accounts_path).resolve()),
        source_format = "multi",
        n_accounts    = len(accounts_df) if accounts_df is not None else 0,
        n_posts       = len(posts_df)    if posts_df    is not None else 0,
    )


def load_bundle(
    input_path: str | Path,
    posts_path:  Optional[str | Path] = None,
    edges_path:  Optional[str | Path] = None,
    labels_path: Optional[str | Path] = None,
    mode: str = "auto",
    nrows: Optional[int] = None,
    adapter: str = "auto",
) -> DataBundle:
    """
    Point d'entree principal recommande.

    Modes originaux :
        "auto"     — detecte automatiquement (un seul fichier csv/json)
        "accounts" — force interpretation comme fichier de comptes
        "posts"    — force interpretation comme fichier de posts
        "multi"    — utilise plusieurs fichiers (posts_path, edges_path, ...)

    Args:
        input_path  : fichier ou DOSSIER principal (ex: dossier twibot-22)
        posts_path  : fichier posts (mode multi uniquement)
        edges_path  : fichier edges (optionnel)
        labels_path : fichier labels (optionnel)
        mode        : "auto" | "accounts" | "posts" | "multi"
        nrows       : limite de lignes
        adapter     : Nom de l'adaptateur ("auto", "flat-file", "twibot-22")
    """
    # 1. Résolution via le Framework d'Adaptateurs
    from src.data.adapters.registry import get_adapter
    
    selected_adapter = get_adapter(adapter_name=adapter, path=input_path)
    
    if selected_adapter.name != "flat-file":
        logger.info(f"Délégation de l'ingestion à l'adaptateur: {selected_adapter.name}")
        return selected_adapter.load(
            base_path=input_path, 
            nrows=nrows, 
            posts_path=posts_path, 
            edges_path=edges_path, 
            labels_path=labels_path
        )

    # 2. Fallback au Legacy Loader (FlatFile interne)
    logger.info("Délégation de l'ingestion à l'adaptateur: flat-file (legacy)")
    if mode == "multi" or (posts_path or edges_path or labels_path):
        return load_multi(
            accounts_path = input_path,
            posts_path    = posts_path,
            edges_path    = edges_path,
            labels_path   = labels_path,
            nrows         = nrows,
        )
    if mode == "accounts":
        return load_accounts(input_path, nrows=nrows)
    if mode == "posts":
        return load_posts(input_path, nrows=nrows)
    return load_file(input_path, nrows=nrows)   # mode == "auto"
