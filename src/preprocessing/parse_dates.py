# -*- coding: utf-8 -*-
"""
src/preprocessing/parse_dates.py
----------------------------------
Parsing des dates et extraction de features temporelles de base.

Principe :
    - Parser toutes les dates en UTC (timezone-safe)
    - Détecter la granularité réelle : full / date_only / coarse / none
    - Extraire : heure, minute, jour, jour de semaine, semaine, mois, week-end, nuit
    - Produire un flag `temporal_granularity` exploitable par le profiler

Fonctions publiques :
    parse_date_column(df, col)        → DataFrame avec col en datetime
    extract_temporal_features(df, col) → DataFrame avec colonnes temporelles dérivées
    detect_granularity(series)         → "full" | "date_only" | "coarse" | "none"
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Seuil pour décider si une série contient de VRAIES heures
# (si >= 5% des timestamps ont une heure > 0, on considère la granularité "full")
_HOUR_PRESENCE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Détection de la granularité
# ---------------------------------------------------------------------------

def detect_granularity(series: pd.Series) -> str:
    """
    Analyse une série de timestamps et retourne la granularité temporelle.

    Returns:
        "full"      → datetime précis avec heures/minutes/secondes
        "date_only" → seulement la date (heures toujours à 00:00:00)
        "coarse"    → approximatif ou partiellement parsable
        "none"      → pas de données temporelles valides
    """
    if series is None or series.empty:
        return "none"

    # Tente de parser si pas encore en datetime
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            parsed = pd.to_datetime(series, utc=True, errors="coerce")
        except Exception:
            return "none"
    else:
        parsed = series

    valid = parsed.dropna()
    if valid.empty:
        return "none"

    # Vérifie si les heures sont non-triviales
    has_hour   = (valid.dt.hour   != 0).mean() >= _HOUR_PRESENCE_THRESHOLD
    has_minute = (valid.dt.minute != 0).mean() >= 0.02

    if has_hour or has_minute:
        return "full"

    # Vérifie si au moins les dates sont cohérentes (pas juste 1970-01-01 partout)
    n_unique_dates = valid.dt.date.nunique()
    if n_unique_dates >= 2:
        return "date_only"

    return "coarse"


# ---------------------------------------------------------------------------
# Parsing de base
# ---------------------------------------------------------------------------

def parse_date_column(
    df: pd.DataFrame,
    col: str,
    utc: bool = True,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    Parse une colonne de dates en datetime Pandas (timezone-safe).

    Args:
        df     : DataFrame source
        col    : nom de la colonne à parser
        utc    : forcer UTC
        errors : "coerce" (NaT si invalide) ou "raise"

    Returns:
        DataFrame avec la colonne convertie en datetime64[ns, UTC]
    """
    if col not in df.columns:
        return df

    if pd.api.types.is_datetime64_any_dtype(df[col]):
        # Déjà en datetime, juste s'assurer qu'il est UTC
        try:
            if df[col].dt.tz is None and utc:
                df = df.copy()
                df[col] = df[col].dt.tz_localize("UTC")
            elif utc:
                df = df.copy()
                df[col] = df[col].dt.tz_convert("UTC")
        except Exception as e:
            logger.warning("Conversion UTC de '%s' échouée : %s", col, e)
        return df

    df = df.copy()
    try:
        df[col] = pd.to_datetime(df[col], utc=utc, errors=errors)
    except Exception as e:
        logger.warning("Parsing de la colonne '%s' échoué : %s", col, e)
        # Tentative de fallback : infer_datetime_format
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True,
                                     utc=utc, errors=errors)
        except Exception:
            df[col] = pd.NaT

    n_valid = df[col].notna().sum()
    n_total = len(df)
    logger.debug("Colonne '%s' parsée : %d/%d valides", col, n_valid, n_total)

    return df


# ---------------------------------------------------------------------------
# Extraction des features temporelles
# ---------------------------------------------------------------------------

def extract_temporal_features(
    df: pd.DataFrame,
    col: str,
    prefix: Optional[str] = None,
    granularity: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extrait les composantes temporelles d'une colonne datetime.

    Features créées (selon la granularité) :

        Toujours (si date disponible) :
            {prefix}_year          — année
            {prefix}_month         — mois (1–12)
            {prefix}_day           — jour du mois (1–31)
            {prefix}_weekday       — jour semaine (0=lundi, 6=dimanche)
            {prefix}_is_weekend    — bool (samedi ou dimanche)
            {prefix}_week_of_year  — semaine ISO (1–53)

        Si granularité "full" :
            {prefix}_hour          — heure (0–23)
            {prefix}_minute        — minute (0–59)
            {prefix}_is_night      — bool (23h–6h)
            {prefix}_hour_bin      — bin de 4h (0, 4, 8, 12, 16, 20)

    Args:
        df          : DataFrame avec la colonne datetime parsée
        col         : nom de la colonne datetime
        prefix      : préfixe des colonnes dérivées (défaut : col sans "_at")
        granularity : granularité forcée (None → auto-détectée)

    Returns:
        DataFrame avec colonnes temporelles supplémentaires
    """
    if col not in df.columns:
        logger.warning("Colonne '%s' absente — extraction temporelle ignorée", col)
        return df

    # Assure que la colonne est en datetime
    df = parse_date_column(df, col)

    ts = df[col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        logger.warning("Colonne '%s' non convertible en datetime", col)
        return df

    # Préfixe des nouvelles colonnes
    if prefix is None:
        prefix = col.replace("_at", "").replace("_time", "")

    df = df.copy()

    # --- Granularité ---
    if granularity is None:
        granularity = detect_granularity(ts)

    df[f"{prefix}_granularity"] = granularity

    # --- Features de date (toujours) ---
    df[f"{prefix}_year"]         = ts.dt.year.astype("Int16")
    df[f"{prefix}_month"]        = ts.dt.month.astype("Int8")
    df[f"{prefix}_day"]          = ts.dt.day.astype("Int8")
    df[f"{prefix}_weekday"]      = ts.dt.weekday.astype("Int8")   # 0=lundi
    df[f"{prefix}_is_weekend"]   = ts.dt.weekday >= 5
    df[f"{prefix}_week_of_year"] = ts.dt.isocalendar().week.astype("Int8")

    # --- Features d'heure (seulement si full) ---
    if granularity == "full":
        df[f"{prefix}_hour"]    = ts.dt.hour.astype("Int8")
        df[f"{prefix}_minute"]  = ts.dt.minute.astype("Int8")
        df[f"{prefix}_is_night"]= (ts.dt.hour >= 23) | (ts.dt.hour <= 6)
        df[f"{prefix}_hour_bin"]= (ts.dt.hour // 4 * 4).astype("Int8")  # 0,4,8,12,16,20

    logger.debug("Features temporelles extraites de '%s' : granularité=%s", col, granularity)
    return df


# ---------------------------------------------------------------------------
# Pipeline complet de preprocessing temporel
# ---------------------------------------------------------------------------

def preprocess_dates(
    df: pd.DataFrame,
    date_columns: Optional[list] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline complet : parse + extrait toutes les colonnes de dates connues.

    Args:
        df           : DataFrame (post normalisation)
        date_columns : liste de colonnes à traiter (None → auto-detect)

    Returns:
        (df_enrichi, granularity_map)
        granularity_map : {col_name → granularité}
    """
    if date_columns is None:
        # Auto-détection des colonnes datetime ou avec alias de date connus
        date_keywords = ("created_at", "updated_at", "timestamp", "date", "time")
        date_columns  = [
            c for c in df.columns
            if any(kw in c.lower() for kw in date_keywords)
            or pd.api.types.is_datetime64_any_dtype(df[c])
        ]

    granularity_map: dict = {}

    for col in date_columns:
        if col not in df.columns:
            continue
        df = parse_date_column(df, col)
        gran = detect_granularity(df[col])
        granularity_map[col] = gran
        df = extract_temporal_features(df, col, granularity=gran)

    return df, granularity_map
