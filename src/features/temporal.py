# -*- coding: utf-8 -*-
"""
src/features/temporal.py
--------------------------
Extracteur de features temporelles — Plan A (précis) et Plan B (dégradé).

Règle essentielle :
    Ce module ne plante JAMAIS. Si les données sont insuffisantes, il
    active le Plan B. Si le Plan B échoue aussi, il retourne un ensemble
    de features nulles plutôt que de propager une exception.

Plan A — timestamps précis (granularité "full") :
    ipt_mean, ipt_std, ipt_cv, hour_entropy, weekday_entropy,
    night_ratio, burst_score, max_gap, sleep_gap, active_hours_count,
    posts_per_hour_max, activity_span_hours

Plan B — timestamps faibles (granularité "date_only" ou "coarse") :
    posts_per_day_mean, posts_per_day_std, n_active_days,
    weekend_ratio, activity_density, days_span

Usage :
    from src.features.temporal import extract_temporal_features

    # Automatique (détecte le plan)
    feat_df = extract_temporal_features(posts_df, accounts_df)

    # Forcé
    feat_df = extract_temporal_features(posts_df, plan="A")
    feat_df = extract_temporal_features(posts_df, plan="B")
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import AccountCols, PostCols
from src.preprocessing.parse_dates import detect_granularity

logger = logging.getLogger(__name__)

# Colonnes Plan A — préfixe "t_"
PLAN_A_COLS = [
    "t_ipt_mean",              # inter-posting time moyen (secondes)
    "t_ipt_std",               # écart-type IPT
    "t_ipt_cv",                # coefficient de variation IPT (std/mean)
    "t_ipt_min",               # IPT minimum
    "t_ipt_max",               # IPT maximum = max gap entre posts
    "t_night_ratio",           # proportion de posts entre 23h et 6h
    # V1.1 — activité nuit fine
    "t_deep_night_ratio",      # proportion de posts entre 0h et 3h (bots)
    "t_late_night_ratio",      # proportion de posts entre 3h et 6h
    "t_hour_entropy",          # entropie de Shannon sur les heures de posts
    "t_weekday_entropy",       # entropie sur les jours de la semaine
    "t_burst_score",           # fraction de posts en burst (<60s d'intervalle)
    # V1.1 — densité de bursts
    "t_burst_density",         # nb moyen de posts dans les fenêtres de 60s
    "t_ipt_log_mean",          # log(1 + ipt_mean) pour stabiliser la distribution
    "t_sleep_gap",             # plus longue pause (heures estimées)
    "t_active_hours_count",    # nb d'heures distinctes avec au moins 1 post
    "t_posts_per_hour_max",    # max de posts dans une fenêtre d'1 heure
    "t_activity_span_hours",   # durée totale d'activité (premier au dernier post)
    # V1.1 — régularité inter-journée
    "t_interday_regularity",   # posts_per_day_std / (posts_per_day_mean + 1)
    "t_active_day_ratio",      # n_active_days / days_span
    "t_n_posts",               # nb total de posts utilisé pour le calcul
]

# Colonnes Plan B — préfixe "t_"
PLAN_B_COLS = [
    "t_posts_per_day_mean",  # posts par jour moyen
    "t_posts_per_day_std",   # écart-type des posts par jour
    "t_n_active_days",       # nb de jours distincts avec activité
    "t_weekend_ratio",       # fraction de posts le week-end
    "t_activity_density",    # n_active_days / days_span (régularité)
    "t_days_span",           # durée totale (jours depuis premier au dernier)
    # V1.1
    "t_interday_regularity", # posts_per_day_std / (posts_per_day_mean + 1)
    "t_active_day_ratio",    # n_active_days / days_span
    "t_n_posts",             # nb total de posts
]


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _shannon_entropy(series: pd.Series, n_bins: int) -> float:
    """Entropie de Shannon normalisée (0=uniforme, 1=concentré)."""
    counts = series.value_counts(normalize=True)
    if counts.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = -float((counts * np.log2(counts + 1e-12)).sum())
    max_h = np.log2(n_bins) if n_bins > 1 else 1.0
    return min(float(h / max_h), 1.0) if max_h > 0 else 0.0


def _null_plan_a(n_posts: int = 0) -> dict:
    """Retourne un dict Plan A avec toutes les features nulles."""
    d = {c: np.nan for c in PLAN_A_COLS}
    d["t_n_posts"] = n_posts
    return d


def _null_plan_b(n_posts: int = 0) -> dict:
    """Retourne un dict Plan B avec toutes les features nulles."""
    d = {c: np.nan for c in PLAN_B_COLS}
    d["t_n_posts"] = n_posts
    return d


# ---------------------------------------------------------------------------
# Plan A — IPT et signaux comportementaux précis
# ---------------------------------------------------------------------------

def _compute_plan_a(posts: pd.DataFrame) -> dict:
    """
    Calcule les features Plan A pour un groupe de posts d'un même compte.

    Args:
        posts : DataFrame filtré pour un compte, trié par timestamp

    Returns:
        dict de features
    """
    n = len(posts)
    if n == 0:
        return _null_plan_a(0)

    col = PostCols.CREATED_AT
    if col not in posts.columns:
        return _null_plan_a(n)

    ts = posts[col].dropna().sort_values()
    n_valid = len(ts)

    if n_valid < 2:
        return _null_plan_a(n)

    # --- Inter-posting times (secondes) ---
    ipt = ts.diff().dropna().dt.total_seconds()
    ipt = ipt[ipt >= 0]   # ignorer les négatifs (désordre résiduel)

    ipt_mean = float(ipt.mean()) if not ipt.empty else np.nan
    ipt_std  = float(ipt.std())  if len(ipt) > 1 else 0.0
    ipt_cv   = float(ipt_std / (ipt_mean + 1e-9)) if ipt_mean and ipt_mean > 0 else np.nan
    ipt_min  = float(ipt.min()) if not ipt.empty else np.nan
    ipt_max  = float(ipt.max()) if not ipt.empty else np.nan

    # --- Night activity (23h–6h) ---
    hours = ts.dt.hour
    night_mask  = (hours >= 23) | (hours <= 6)
    night_ratio = float(night_mask.mean())

    # ── V1.1 — Activité nuit fine ─────────────────────────────────────────
    deep_night_mask = (hours >= 0) & (hours < 3)
    late_night_mask = (hours >= 3) & (hours < 6)
    deep_night_ratio = float(deep_night_mask.mean())
    late_night_ratio = float(late_night_mask.mean())

    # --- Entropie des heures (0-23) ---
    hour_entropy = _shannon_entropy(hours, n_bins=24)

    # --- Entropie des jours de semaine (0-6) ---
    weekday_entropy = _shannon_entropy(ts.dt.weekday, n_bins=7)

    # --- Burst score : fraction d'IPT < 60s ---
    burst_score = float((ipt < 60).mean()) if not ipt.empty else 0.0

    # ── V1.1 — Densité de bursts ──────────────────────────────────────────
    # Nb moyen de posts consécutifs dans une fenêtre de 60s
    if not ipt.empty:
        burst_windows = (ipt < 60)
        if burst_windows.any():
            # Taille des runs de bursts consécutifs
            run_sizes = []
            count = 1
            for b in burst_windows:
                if b:
                    count += 1
                else:
                    if count > 1:
                        run_sizes.append(count)
                    count = 1
            if count > 1:
                run_sizes.append(count)
            burst_density = float(np.mean(run_sizes)) if run_sizes else 1.0
        else:
            burst_density = 1.0
    else:
        burst_density = np.nan

    # ── V1.1 — Log(1 + ipt_mean) ─────────────────────────────────────────
    ipt_log_mean = float(np.log1p(ipt_mean)) if not np.isnan(ipt_mean) else np.nan

    # --- Sleep gap : plus longue pause en heures ---
    sleep_gap = float(ipt_max / 3600.0) if not np.isnan(ipt_max) else np.nan

    # --- Heures distinctes avec activité ---
    active_hours = int(hours.nunique())

    # --- Max posts dans une fenêtre de 1 heure ---
    try:
        ts_series = ts.reset_index(drop=True)
        ts_unix   = ts_series.astype(np.int64) / 1e9
        counts_in_window = [
            int(((ts_unix >= t) & (ts_unix < t + 3600)).sum())
            for t in ts_unix
        ]
        posts_per_hour_max = int(max(counts_in_window)) if counts_in_window else 0
    except Exception:
        posts_per_hour_max = 0

    # --- Durée totale d'activité en heures ---
    span_seconds   = float((ts.iloc[-1] - ts.iloc[0]).total_seconds())
    activity_span  = span_seconds / 3600.0

    # ── V1.1 — Régularité inter-journée ──────────────────────────────────
    try:
        dates          = ts.dt.date
        date_counts    = pd.Series(dates).value_counts()
        ppd_mean       = float(date_counts.mean())
        ppd_std        = float(date_counts.std()) if len(date_counts) > 1 else 0.0
        days_span      = max(int((dates.max() - dates.min()).days) + 1, 1)
        n_active_days  = len(date_counts)
        interday_reg   = round(ppd_std / (ppd_mean + 1), 4)
        active_day_r   = round(n_active_days / days_span, 4)
    except Exception:
        interday_reg   = np.nan
        active_day_r   = np.nan

    return {
        "t_ipt_mean":            round(ipt_mean, 4)         if not np.isnan(ipt_mean) else np.nan,
        "t_ipt_std":             round(ipt_std, 4),
        "t_ipt_cv":              round(ipt_cv, 4)           if not np.isnan(ipt_cv)   else np.nan,
        "t_ipt_min":             round(ipt_min, 4)          if not np.isnan(ipt_min)  else np.nan,
        "t_ipt_max":             round(ipt_max, 4)          if not np.isnan(ipt_max)  else np.nan,
        "t_night_ratio":         round(night_ratio, 4),
        "t_deep_night_ratio":    round(deep_night_ratio, 4),
        "t_late_night_ratio":    round(late_night_ratio, 4),
        "t_hour_entropy":        round(hour_entropy, 4),
        "t_weekday_entropy":     round(weekday_entropy, 4),
        "t_burst_score":         round(burst_score, 4),
        "t_burst_density":       round(burst_density, 4)    if not np.isnan(burst_density) else np.nan,
        "t_ipt_log_mean":        round(ipt_log_mean, 4)     if not np.isnan(ipt_log_mean) else np.nan,
        "t_sleep_gap":           round(sleep_gap, 4)        if not np.isnan(sleep_gap)    else np.nan,
        "t_active_hours_count":  active_hours,
        "t_posts_per_hour_max":  posts_per_hour_max,
        "t_activity_span_hours": round(activity_span, 4),
        "t_interday_regularity": interday_reg,
        "t_active_day_ratio":    active_day_r,
        "t_n_posts":             n_valid,
    }


# ---------------------------------------------------------------------------
# Plan B — features agrégées (dates sans heure)
# ---------------------------------------------------------------------------

def _compute_plan_b(posts: pd.DataFrame) -> dict:
    """
    Calcule les features Plan B pour un groupe de posts d'un même compte.

    Args:
        posts : DataFrame filtré pour un compte

    Returns:
        dict de features
    """
    n = len(posts)
    if n == 0:
        return _null_plan_b(0)

    col = PostCols.CREATED_AT
    if col not in posts.columns:
        return _null_plan_b(n)

    ts = posts[col].dropna().sort_values()
    if ts.empty:
        return _null_plan_b(n)

    # Normalise en date (supprime l'heure si présente)
    try:
        dates = ts.dt.date
    except Exception:
        return _null_plan_b(n)

    date_counts = pd.Series(dates).value_counts().sort_index()
    n_active_days = int(date_counts.shape[0])

    posts_per_day_mean = float(date_counts.mean()) if n_active_days > 0 else 0.0
    posts_per_day_std  = float(date_counts.std())  if n_active_days > 1 else 0.0

    # Durée totale en jours
    try:
        days_span = int((dates.max() - dates.min()).days) + 1
    except Exception:
        days_span = n_active_days

    # Densité : nb de jours actifs / durée totale
    activity_density = float(n_active_days / max(days_span, 1))

    # Weekend ratio
    try:
        weekday_vals  = ts.dt.weekday
        weekend_ratio = float((weekday_vals >= 5).mean())
    except Exception:
        weekend_ratio = 0.0

    # ── V1.1 — Régularité inter-journée ──────────────────────────────────
    interday_regularity = round(
        float(posts_per_day_std / (posts_per_day_mean + 1)), 4
    )
    active_day_ratio = round(activity_density, 4)   # alias explicite

    return {
        "t_posts_per_day_mean":  round(posts_per_day_mean, 4),
        "t_posts_per_day_std":   round(posts_per_day_std, 4),
        "t_n_active_days":       n_active_days,
        "t_weekend_ratio":       round(weekend_ratio, 4),
        "t_activity_density":    round(activity_density, 4),
        "t_days_span":           days_span,
        "t_interday_regularity": interday_regularity,
        "t_active_day_ratio":    active_day_ratio,
        "t_n_posts":             n,
    }


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def extract_temporal_features(
    posts_df:    pd.DataFrame,
    accounts_df: Optional[pd.DataFrame] = None,
    plan:        Optional[str] = None,
    min_posts:   int = 2,
) -> pd.DataFrame:
    """
    Extrait les features temporelles pour chaque compte.

    Détecte automatiquement le plan (A ou B) sauf si `plan` est forcé.

    Args:
        posts_df    : DataFrame de posts avec account_id et created_at
        accounts_df : (optionnel) pour créer les lignes des comptes sans posts
        plan        : "A", "B", ou None (auto-détection)
        min_posts   : nb minimal de posts pour calculer les features Plan A (sinon Plan B)

    Returns:
        DataFrame avec account_id + features temporelles
        Chaque compte a exactement une ligne.
        Les comptes sans posts reçoivent des NaN.
    """
    if posts_df is None or posts_df.empty:
        logger.warning("posts_df vide — features temporelles nulles")
        return pd.DataFrame(columns=[AccountCols.ID] + PLAN_B_COLS)

    id_col = AccountCols.ID
    ts_col = PostCols.CREATED_AT

    # --- Détection du plan ---
    if plan is None:
        if ts_col in posts_df.columns:
            gran = detect_granularity(posts_df[ts_col])
            plan = "A" if gran == "full" else "B"
        else:
            plan = "B"
    plan = plan.upper()
    logger.info("Module temporel : Plan %s activé", plan)

    # --- Parsing des dates si nécessaire ---
    if ts_col in posts_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(posts_df[ts_col]):
            posts_df = posts_df.copy()
            posts_df[ts_col] = pd.to_datetime(posts_df[ts_col], utc=True, errors="coerce")

    # --- Calcul par compte ---
    if id_col not in posts_df.columns:
        logger.warning("Colonne account_id absente dans posts_df")
        return pd.DataFrame(columns=[id_col] + (PLAN_A_COLS if plan == "A" else PLAN_B_COLS))

    compute_fn = _compute_plan_a if plan == "A" else _compute_plan_b
    results = []

    for account_id, group in posts_df.groupby(id_col, sort=False):
        try:
            # Si Plan A mais pas assez de posts → fallback Plan B
            if plan == "A" and len(group) < min_posts:
                feat = _compute_plan_b(group)
                # Compléter les colonnes Plan A manquantes avec NaN
                full = _null_plan_a(len(group))
                full.update(feat)
                feat = full
            else:
                feat = compute_fn(group)
        except Exception as e:
            logger.warning("Erreur temporelle compte '%s': %s", account_id, e)
            feat = _null_plan_a(0) if plan == "A" else _null_plan_b(0)

        feat[id_col] = account_id
        results.append(feat)

    if not results:
        return pd.DataFrame(columns=[id_col] + (PLAN_A_COLS if plan == "A" else PLAN_B_COLS))

    result_df = pd.DataFrame(results)

    # --- Ajouter les comptes sans aucun post (si accounts_df fourni) ---
    if accounts_df is not None and id_col in accounts_df.columns:
        known_ids   = set(result_df[id_col])
        missing_ids = set(accounts_df[id_col]) - known_ids
        if missing_ids:
            null_fn  = _null_plan_a if plan == "A" else _null_plan_b
            null_rows = []
            for aid in missing_ids:
                row = null_fn(0)
                row[id_col] = aid
                null_rows.append(row)
            result_df = pd.concat(
                [result_df, pd.DataFrame(null_rows)],
                ignore_index=True
            )

    # Mise en ordre des colonnes : account_id en premier
    expected_cols = PLAN_A_COLS if plan == "A" else PLAN_B_COLS
    final_cols    = [id_col] + [c for c in expected_cols if c in result_df.columns]
    result_df = result_df[final_cols].reset_index(drop=True)

    logger.info(
        "Features temporelles (Plan %s) : %d comptes × %d features",
        plan, len(result_df), len(final_cols) - 1
    )
    return result_df


# ---------------------------------------------------------------------------
# Alias pour confort d'import
# ---------------------------------------------------------------------------

extract_temporal_plan_a = lambda posts_df, **kw: extract_temporal_features(posts_df, plan="A", **kw)
extract_temporal_plan_b = lambda posts_df, **kw: extract_temporal_features(posts_df, plan="B", **kw)
