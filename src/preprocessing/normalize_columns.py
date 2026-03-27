# -*- coding: utf-8 -*-
"""
src/preprocessing/normalize_columns.py
---------------------------------------
Normalisation des noms de colonnes brutes vers les noms canoniques du pipeline.

Règle centrale :
    Après normalisation, AUCUN autre module ne doit connaître les noms bruts.
    Tout le pipeline travaille exclusivement avec les noms définis dans schema.py.

Usage :
    from src.preprocessing.normalize_columns import normalize_columns, ColumnReport

    df, report = normalize_columns(df)
    print(report.summary())
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.data.schema import (
    ACCOUNT_ID_ALIASES,
    BIO_ALIASES,
    CREATED_AT_ALIASES,
    FOLLOWERS_ALIASES,
    FOLLOWING_ALIASES,
    LABEL_ALIASES,
    POST_ID_ALIASES,
    SCREEN_NAME_ALIASES,
    SOURCE_ALIASES,
    TEXT_ALIASES,
    TOTAL_POSTS_ALIASES,
    AccountCols,
    LabelCols,
    PostCols,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rapport de normalisation
# ---------------------------------------------------------------------------

@dataclass
class ColumnReport:
    """
    Rapport produit après normalisation.
    Permet de tracer exactement ce qui a été renommé.
    """
    renamed:    Dict[str, str] = field(default_factory=dict)   # brut → canonique
    kept:       List[str]      = field(default_factory=list)   # inchangées
    dropped:    List[str]      = field(default_factory=list)   # colonnes dupliquées supprimées
    unknown:    List[str]      = field(default_factory=list)   # colonnes sans alias connu
    normalized: List[str]      = field(default_factory=list)   # liste finale des colonnes

    def summary(self) -> str:
        lines = ["ColumnReport :"]
        if self.renamed:
            lines.append("  Renommages :")
            for src, dst in self.renamed.items():
                lines.append(f"    '{src}' → '{dst}'")
        if self.dropped:
            lines.append(f"  Doublons supprimes : {self.dropped}")
        if self.unknown:
            lines.append(f"  Colonnes inconnues (conservees) : {self.unknown}")
        lines.append(f"  Colonnes finales : {self.normalized}")
        return "\n".join(lines)

    @property
    def n_renamed(self) -> int:
        return len(self.renamed)


# ---------------------------------------------------------------------------
# Table de mapping étendue (alias → colonne canonique)
# ---------------------------------------------------------------------------

def _build_alias_table() -> Dict[str, str]:
    """
    Construit le dictionnaire complet alias_lowercase → colonne_canonique.
    Priorité : les mappings plus spécifiques d'abord.
    """
    table: Dict[str, str] = {}

    groups = [
        (ACCOUNT_ID_ALIASES,  AccountCols.ID),
        (POST_ID_ALIASES,     PostCols.ID),
        (TEXT_ALIASES,        PostCols.TEXT),
        (CREATED_AT_ALIASES,  PostCols.CREATED_AT),  # canonical sera override si accounts
        (SCREEN_NAME_ALIASES, AccountCols.SCREEN_NAME),
        (BIO_ALIASES,         AccountCols.BIO),
        (FOLLOWERS_ALIASES,   AccountCols.FOLLOWERS),
        (FOLLOWING_ALIASES,   AccountCols.FOLLOWING),
        (TOTAL_POSTS_ALIASES, AccountCols.TOTAL_POSTS),
        (LABEL_ALIASES,       LabelCols.LABEL),
        (SOURCE_ALIASES,      PostCols.SOURCE),
    ]

    for aliases, canonical in groups:
        for alias in aliases:
            key = alias.lower().strip()
            if key not in table:           # premier match gagne
                table[key] = canonical

    return table


# Singleton construit une seule fois au chargement du module
_ALIAS_TABLE: Dict[str, str] = _build_alias_table()


# ---------------------------------------------------------------------------
# Nettoyage des noms bruts avant matching
# ---------------------------------------------------------------------------

def _sanitize_col_name(name: str) -> str:
    """
    Normalise un nom de colonne brut avant tentative de matching :
      - strip des espaces
      - lowercase
      - remplacement des espaces/tirets par underscores
      - suppression des accents courants
    """
    s = name.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    # Suppression d'accents courants pour robustesse
    accent_map = str.maketrans("àâäéèêëîïôùûüç", "aaaeeeeiioouuc")
    s = s.translate(accent_map)
    return s


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def normalize_columns(
    df: pd.DataFrame,
    extra_aliases: Optional[Dict[str, str]] = None,
    drop_duplicates: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, ColumnReport]:
    """
    Normalise les noms de colonnes d'un DataFrame vers les noms canoniques.

    Stratégie (dans l'ordre) :
      1. Sanitize le nom brut (lowercase, underscores, sans accents)
      2. Cherche dans la table d'alias complète
      3. Si trouvé → renomme
      4. Si la colonne canonique existe déjà → marque la brute comme doublon
      5. Si aucun alias → conserve tel quel (marqué "unknown")
      6. Supprime les doublons si drop_duplicates=True

    Args:
        df              : DataFrame brut entrant
        extra_aliases   : mapping supplémentaire {alias_lowercase → canonique}
                          pour surcharger / étendre les aliases par défaut
        drop_duplicates : supprimer les colonnes dupliquées après renommage
        verbose         : logger le rapport complet

    Returns:
        (df_normalisé, ColumnReport)
    """
    report = ColumnReport()
    alias_table = dict(_ALIAS_TABLE)

    # Aliases supplémentaires (prioritaires)
    if extra_aliases:
        alias_table.update({k.lower().strip(): v for k, v in extra_aliases.items()})

    rename_map: Dict[str, str] = {}
    current_canonicals: set = set()    # colonnes canoniques déjà présentes dans df

    # Pré-indexer les colonnes actuelles
    existing_cols = set(df.columns)

    for col in df.columns:
        sanitized = _sanitize_col_name(col)
        canonical = alias_table.get(sanitized)

        if canonical is None:
            # Pas d'alias connu → colonne inconnue conservée
            report.unknown.append(col)
            report.kept.append(col)
            continue

        if col == canonical:
            # Déjà au bon nom
            report.kept.append(col)
            current_canonicals.add(canonical)
            continue

        if canonical in existing_cols or canonical in rename_map.values():
            # La colonne canonique existe déjà → ce sera un doublon
            report.dropped.append(col)
            continue

        # Renommage valide
        rename_map[col] = canonical
        report.renamed[col] = canonical
        current_canonicals.add(canonical)

    # Appliquer les renommages
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug("Colonnes renommees : %s", rename_map)

    # Supprimer les doublons (colonnes marquées dropped)
    if drop_duplicates and report.dropped:
        cols_to_drop = [c for c in report.dropped if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.debug("Colonnes doublons supprimees : %s", cols_to_drop)

    report.normalized = list(df.columns)

    if verbose:
        logger.info(report.summary())

    return df, report


# ---------------------------------------------------------------------------
# Utilitaires complémentaires
# ---------------------------------------------------------------------------

def get_canonical_name(raw_name: str, extra_aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Retourne le nom canonique correspondant à un alias, ou None si inconnu.

    Args:
        raw_name      : nom de colonne brut
        extra_aliases : mapping supplémentaire optionnel

    Returns:
        Nom canonique ou None
    """
    sanitized = _sanitize_col_name(raw_name)
    alias_table = dict(_ALIAS_TABLE)
    if extra_aliases:
        alias_table.update({k.lower().strip(): v for k, v in extra_aliases.items()})
    return alias_table.get(sanitized)


def list_all_aliases() -> Dict[str, str]:
    """
    Retourne une copie de la table complète alias → canonique.
    Utile pour le debugging et l'inspection.
    """
    return dict(_ALIAS_TABLE)


def check_required_columns(
    df: pd.DataFrame,
    required: List[str],
) -> tuple[bool, List[str]]:
    """
    Vérifie qu'un DataFrame normalisé contient toutes les colonnes requises.

    Args:
        df       : DataFrame (après normalize_columns)
        required : liste de noms canoniques requis

    Returns:
        (ok, missing_list)
    """
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing


def normalize_and_validate(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    extra_aliases: Optional[Dict[str, str]] = None,
) -> tuple[pd.DataFrame, ColumnReport, bool, List[str]]:
    """
    Normalise + valide qu'un ensemble de colonnes requises est présent.

    Args:
        df               : DataFrame brut
        required_columns : colonnes canoniques requises (None = pas de vérification)
        extra_aliases    : aliases supplémentaires

    Returns:
        (df_normalisé, rapport, ok, colonnes_manquantes)
    """
    df, report = normalize_columns(df, extra_aliases=extra_aliases)
    if required_columns:
        ok, missing = check_required_columns(df, required_columns)
    else:
        ok, missing = True, []
    return df, report, ok, missing
