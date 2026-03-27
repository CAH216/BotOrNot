# -*- coding: utf-8 -*-
"""
src/data/profiler.py
--------------------
Profiler automatique du dataset BotOrNot.

Il analyse un DataBundle et produit un DataProfile contenant :
  - des flags booléens  (has_text, has_timestamps, ...)
  - des métriques de qualité (n_accounts, class_balance, ...)
  - des recommandations de modules à activer

C'est le cerveau du pipeline : il décide quels extracteurs tourneront.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.schema import AccountCols, DataBundle, LabelCols, PostCols

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Résultat du profiler
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """
    Conteneur des flags et métriques produits par le profiler.
    Circule dans le pipeline à côté du DataBundle.
    """

    # --- Flags principaux ---
    has_text:                bool = False   # au moins une colonne texte disponible
    has_timestamps:          bool = False   # timestamps présents (précision quelconque)
    has_precise_timestamps:  bool = False   # timestamps avec heure (pas juste date)
    has_labels:              bool = False   # labels de classification disponibles
    has_account_ids:         bool = False   # identifiants de comptes présents
    has_post_ids:            bool = False   # identifiants de posts présents
    has_edges:               bool = False   # données relationnelles (edges_df)
    has_profile_metadata:    bool = False   # metadata du profil (followers, bio, etc.)
    has_structural_signals:  bool = False   # signaux structurels (source, API, IDs anormaux)
    has_multilingual_text:   bool = False   # textes en plusieurs langues détectées

    # --- Granularité temporelle ---
    # "full"      → datetime complet avec heures/minutes/secondes
    # "date_only" → seulement date (YYYY-MM-DD)
    # "coarse"    → approximatif (semaine, mois)
    # "none"      → pas de timestamps
    temporal_granularity: str = "none"

    # --- Métriques quantitatives ---
    n_accounts:            int   = 0
    n_posts:               int   = 0
    n_edges:               int   = 0
    n_labels:              int   = 0
    posts_per_account_mean: float = 0.0
    posts_per_account_max:  int   = 0

    # --- Déséquilibre de classes ---
    class_balance:    Optional[Dict[str, Any]] = None
    imbalance_ratio:  Optional[float]          = None   # majority / minority

    # --- Qualité des données ---
    missing_rates:    Dict[str, float] = field(default_factory=dict)
    text_empty_rate:  float = 0.0     # proportion de posts avec texte vide
    label_noise_rate: float = 0.0     # proportion de labels non reconnus

    # --- Modules recommandés ---
    recommended_modules: List[str] = field(default_factory=list)
    disabled_modules:    List[str] = field(default_factory=list)
    warnings:            List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Retourne les flags principaux sous forme de dictionnaire."""
        return {
            "has_text":               self.has_text,
            "has_timestamps":         self.has_timestamps,
            "has_precise_timestamps": self.has_precise_timestamps,
            "has_labels":             self.has_labels,
            "has_account_ids":        self.has_account_ids,
            "has_post_ids":           self.has_post_ids,
            "has_edges":              self.has_edges,
            "has_profile_metadata":   self.has_profile_metadata,
            "has_structural_signals": self.has_structural_signals,
            "has_multilingual_text":  self.has_multilingual_text,
            "temporal_granularity":   self.temporal_granularity,
        }

    def summary(self) -> str:
        lines = ["DataProfile :"]
        flags = self.to_dict()
        for k, v in flags.items():
            icon = "+" if v not in (False, "none") else "-"
            lines.append(f"  [{icon}] {k:<30} = {v}")
        if self.recommended_modules:
            lines.append(f"  Modules recommandés : {', '.join(self.recommended_modules)}")
        if self.disabled_modules:
            lines.append(f"  Modules désactivés  : {', '.join(self.disabled_modules)}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fonctions d'analyse internes
# ---------------------------------------------------------------------------

def _check_timestamps(posts_df: pd.DataFrame) -> tuple[bool, bool, str]:
    """
    Vérifie la disponibilité et la précision des timestamps dans posts_df.

    Returns:
        (has_timestamps, has_precise_timestamps, temporal_granularity)
    """
    col = PostCols.CREATED_AT
    if col not in posts_df.columns:
        return False, False, "none"

    ts = posts_df[col].dropna()
    if ts.empty:
        return False, False, "none"

    has_ts = True

    # Vérifie si c'est déjà en datetime
    if pd.api.types.is_datetime64_any_dtype(ts):
        # Regarde si les heures sont non-nulles (vrais timestamps avec heure)
        has_hour = (ts.dt.hour != 0).any() or (ts.dt.minute != 0).any()
        if has_hour:
            return True, True, "full"
        else:
            return True, False, "date_only"

    # Tente de parser si c'est du texte
    try:
        parsed = pd.to_datetime(ts, utc=True, errors="coerce")
        valid = parsed.dropna()
        if valid.empty:
            return True, False, "coarse"
        has_hour = (valid.dt.hour != 0).any() or (valid.dt.minute != 0).any()
        if has_hour:
            return True, True, "full"
        return True, False, "date_only"
    except Exception:
        return True, False, "coarse"


def _check_account_timestamps(accounts_df: pd.DataFrame) -> tuple[bool, bool, str]:
    """Même vérification mais sur la colonne created_at des comptes."""
    col = AccountCols.CREATED_AT
    if col not in accounts_df.columns:
        return False, False, "none"
    ts = accounts_df[col].dropna()
    if ts.empty:
        return False, False, "none"
    if pd.api.types.is_datetime64_any_dtype(ts):
        has_hour = (ts.dt.hour != 0).any()
        return True, has_hour, "full" if has_hour else "date_only"
    return True, False, "coarse"


def _check_multilingual(posts_df: pd.DataFrame) -> bool:
    """
    Détecte si le dataset contient du texte multilingue.
    Stratégie légère : colonne lang + heuristique sur les caractères.
    """
    # Vérification via colonne lang
    if PostCols.LANG in posts_df.columns:
        langs = posts_df[PostCols.LANG].dropna().unique()
        if len(langs) > 1:
            return True

    # Heuristique sur les caractères si colonne texte disponible
    if PostCols.TEXT in posts_df.columns:
        texts = posts_df[PostCols.TEXT].dropna().astype(str)
        if texts.empty:
            return False
        sample = texts.head(200)

        # Présence de caractères non-ASCII = probable multilinguisme
        has_non_ascii = sample.str.contains(r"[^\x00-\x7F]", regex=True).mean()
        if has_non_ascii > 0.1:
            return True

        # Présence de caractères arabes, cyrilliques, CJK...
        has_exotic = sample.str.contains(
            r"[\u0600-\u06FF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF]",
            regex=True
        ).mean()
        if has_exotic > 0.05:
            return True

    return False


def _check_structural_signals(accounts_df: pd.DataFrame,
                               posts_df: Optional[pd.DataFrame]) -> bool:
    """
    Détecte la présence de signaux structurels exploitables.
    Signaux : source/API, colonnes ID séquentielles, flags systèmes.
    """
    # Colonne source/client (très utile)
    if posts_df is not None and PostCols.SOURCE in posts_df.columns:
        src = posts_df[PostCols.SOURCE].dropna()
        if len(src) > 0 and src.nunique() > 1:
            return True

    # Colonne verified
    if AccountCols.VERIFIED in accounts_df.columns:
        return True

    # Colonne default_profile_image
    if AccountCols.PROFILE_IMAGE in accounts_df.columns:
        return True

    # IDs numériques dans accounts — potentiellement séquentiels
    if AccountCols.ID in accounts_df.columns:
        ids = accounts_df[AccountCols.ID].dropna()
        try:
            numeric_ids = pd.to_numeric(ids, errors="coerce").dropna()
            if len(numeric_ids) > 5:
                # Vérifie si les IDs sont quasi-séquentiels (batch creation pattern)
                sorted_ids = numeric_ids.sort_values().reset_index(drop=True)
                diffs = sorted_ids.diff().dropna()
                cv = diffs.std() / (diffs.mean() + 1e-9)
                if cv < 0.5:   # faible variation = très séquentiel
                    return True
        except Exception:
            pass

    return False


def _compute_class_balance(labels_df: pd.DataFrame) -> tuple[dict, Optional[float]]:
    """Calcule la distribution et le ratio de déséquilibre."""
    if LabelCols.LABEL not in labels_df.columns:
        return {}, None

    counts = labels_df[LabelCols.LABEL].value_counts()
    total  = len(labels_df)
    balance = {
        str(int(k)): {
            "count": int(v),
            "pct":   round(100 * v / total, 1),
        }
        for k, v in counts.items()
    }

    if len(counts) >= 2:
        ratio = round(counts.iloc[0] / counts.iloc[-1], 2)
    else:
        ratio = None

    return balance, ratio


def _compute_missing_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Retourne le taux de valeurs manquantes par colonne (seulement si > 0)."""
    rates = {}
    for col in df.columns:
        rate = df[col].isna().mean()
        if rate > 0:
            rates[col] = round(float(rate), 4)
    return rates


def _recommend_modules(profile: DataProfile) -> tuple[list, list, list]:
    """
    Déduit les modules à activer / désactiver selon les flags.
    Retourne (recommended, disabled, warnings).
    """
    rec  = []
    dis  = []
    warn = []

    # Tabulaire — presque toujours actif
    if profile.has_account_ids or profile.has_profile_metadata:
        rec.append("tabular")
    else:
        dis.append("tabular")
        warn.append("Pas d'account_ids ni de metadata → module tabulaire limité")

    # Temporel
    if profile.has_precise_timestamps:
        rec.append("temporal_plan_a")
    elif profile.has_timestamps:
        rec.append("temporal_plan_b")
        warn.append("Timestamps sans heure précise → plan B temporel (features agrégées)")
    else:
        dis.append("temporal")
        warn.append("Pas de timestamps → module temporel désactivé")

    # Texte léger
    if profile.has_text:
        rec.append("text_basic")
    else:
        dis.append("text_basic")
        warn.append("Pas de texte → module texte désactivé")

    # Texte profond (optionnel par défaut)
    if profile.has_text and profile.n_posts > 500:
        rec.append("text_embeddings_optional")
    else:
        dis.append("text_embeddings")

    # Structurel
    if profile.has_structural_signals:
        rec.append("structural")
    else:
        dis.append("structural")

    # Relationnel
    if profile.has_edges:
        rec.append("relational")
    else:
        dis.append("relational")

    # Labels
    if not profile.has_labels:
        warn.append("Pas de labels → entraînement impossible, mode inférence uniquement")

    # Déséquilibre
    if profile.imbalance_ratio and profile.imbalance_ratio > 5:
        warn.append(
            f"Déséquilibre fort (ratio {profile.imbalance_ratio:.1f}:1) → "
            "utiliser class_weight + tuning du seuil"
        )

    # Multilinguisme
    if profile.has_multilingual_text:
        warn.append("Texte multilingue détecté → préférer XLM-R si deep NLP activé")

    return rec, dis, warn


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def profile_bundle(bundle: DataBundle) -> DataProfile:
    """
    Analyse un DataBundle et retourne un DataProfile complet.

    Args:
        bundle : DataBundle produit par load_bundle()

    Returns:
        DataProfile avec tous les flags, métriques et recommandations
    """
    profile = DataProfile()

    accounts_df = bundle.accounts_df
    posts_df    = bundle.posts_df
    edges_df    = bundle.edges_df
    labels_df   = bundle.labels_df

    # ---- Comptes ----
    if accounts_df is not None and not accounts_df.empty:
        profile.n_accounts = len(accounts_df)

        profile.has_account_ids = AccountCols.ID in accounts_df.columns

        profile.has_profile_metadata = any(
            c in accounts_df.columns
            for c in (AccountCols.FOLLOWERS, AccountCols.FOLLOWING,
                      AccountCols.BIO, AccountCols.TOTAL_POSTS,
                      AccountCols.VERIFIED, AccountCols.PROFILE_IMAGE)
        )

        # Missing rates sur les comptes
        profile.missing_rates.update(
            {f"accounts.{k}": v
             for k, v in _compute_missing_rates(accounts_df).items()}
        )

        # Timestamps des comptes (date de création)
        acc_has_ts, acc_precise, acc_gran = _check_account_timestamps(accounts_df)
        if acc_has_ts and not profile.has_timestamps:
            profile.has_timestamps = acc_has_ts
            profile.has_precise_timestamps = acc_precise
            profile.temporal_granularity = acc_gran

    # ---- Posts ----
    if posts_df is not None and not posts_df.empty:
        profile.n_posts = len(posts_df)

        profile.has_post_ids = PostCols.ID in posts_df.columns

        profile.has_text = (
            PostCols.TEXT in posts_df.columns
            and posts_df[PostCols.TEXT].notna().any()
        )

        if profile.has_text:
            empty_texts = (
                posts_df[PostCols.TEXT]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .mean()
            )
            profile.text_empty_rate = round(float(empty_texts), 4)

        # Timestamps posts (plus précis que les comptes en général)
        ts_has, ts_precise, ts_gran = _check_timestamps(posts_df)
        # Prend la meilleure granularité disponible
        if ts_has:
            profile.has_timestamps = True
            if ts_precise:
                profile.has_precise_timestamps = True
                profile.temporal_granularity = "full"
            elif profile.temporal_granularity not in ("full",):
                profile.temporal_granularity = ts_gran

        # Multilinguisme
        profile.has_multilingual_text = _check_multilingual(posts_df)

        # Missing rates sur les posts
        profile.missing_rates.update(
            {f"posts.{k}": v
             for k, v in _compute_missing_rates(posts_df).items()}
        )

        # Posts par compte
        if AccountCols.ID in posts_df.columns and profile.n_accounts > 0:
            counts = posts_df[AccountCols.ID].value_counts()
            profile.posts_per_account_mean = round(float(counts.mean()), 2)
            profile.posts_per_account_max  = int(counts.max())

    # ---- Edges ----
    if edges_df is not None and not edges_df.empty:
        profile.has_edges = True
        profile.n_edges   = len(edges_df)

    # ---- Labels ----
    if labels_df is not None and not labels_df.empty:
        profile.has_labels = True
        profile.n_labels   = len(labels_df)

        balance, ratio = _compute_class_balance(labels_df)
        profile.class_balance   = balance
        profile.imbalance_ratio = ratio

        # Bruit dans les labels
        if LabelCols.LABEL in labels_df.columns:
            noise = labels_df[LabelCols.LABEL].isna().mean()
            profile.label_noise_rate = round(float(noise), 4)

    # ---- Signaux structurels ----
    if accounts_df is not None:
        profile.has_structural_signals = _check_structural_signals(
            accounts_df, posts_df
        )

    # ---- Recommandations ----
    rec, dis, warn = _recommend_modules(profile)
    profile.recommended_modules = rec
    profile.disabled_modules    = dis
    profile.warnings            = warn

    logger.info("Profiling terminé : %d flags actifs", sum(profile.to_dict().values().__len__()))
    return profile
