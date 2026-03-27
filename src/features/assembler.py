# -*- coding: utf-8 -*-
"""
src/features/assembler.py
--------------------------
Assembleur central de features du pipeline BotOrNot.

Rôle :
    Fusionner les blocs de features produits par les différents extracteurs
    en une matrice finale unique, alignée par account_id.

Principe de conception :
    - Chaque extracteur produit un DataFrame avec account_id + ses features
    - L'assembleur les joint par LEFT JOIN sur account_id
    - Les blocs manquants sont ignorés (pas de plantage)
    - La matrice finale est reproductible via un FeatureMap sauvegardable
    - Les colonnes manquantes au moment de l'inférence sont recréées avec fill_value

Usage :
    from src.features.assembler import FeatureAssembler

    asm = FeatureAssembler()
    asm.add_block("tabular",    tabular_df)
    asm.add_block("temporal",   temporal_df)
    asm.add_block("text_basic", text_df)          # optionnel
    asm.add_block("structural", structural_df)    # optionnel

    X, feature_map = asm.assemble(labels_df=labels_df)
    asm.save_map("artifacts/feature_maps/v1.json")

    # Inférence
    asm2 = FeatureAssembler.load_map("artifacts/feature_maps/v1.json")
    X_test = asm2.transform(test_blocks)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import AccountCols, LabelCols

logger = logging.getLogger(__name__)

# Valeur de remplissage par défaut pour les NaN numériques
_DEFAULT_FILL = -999.0


# ---------------------------------------------------------------------------
# FeatureMap — descripteur de la matrice finale
# ---------------------------------------------------------------------------

@dataclass
class FeatureMap:
    """
    Décrit la matrice de features : quelles colonnes, dans quel ordre,
    avec quelle valeur par défaut au remplissage.

    Sauvegardé en JSON pour inférence reproductible.
    """
    feature_columns: List[str]           = field(default_factory=list)
    block_columns:   Dict[str, List[str]] = field(default_factory=dict)
    fill_values:     Dict[str, float]     = field(default_factory=dict)
    active_blocks:   List[str]            = field(default_factory=list)
    n_features:      int                  = 0

    def to_dict(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "block_columns":   self.block_columns,
            "fill_values":     self.fill_values,
            "active_blocks":   self.active_blocks,
            "n_features":      self.n_features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureMap":
        return cls(
            feature_columns = d.get("feature_columns", []),
            block_columns   = d.get("block_columns", {}),
            fill_values     = d.get("fill_values", {}),
            active_blocks   = d.get("active_blocks", []),
            n_features      = d.get("n_features", 0),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("FeatureMap sauvegardé : %s (%d features)", path, self.n_features)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureMap":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        lines = [
            f"FeatureMap : {self.n_features} features",
            f"  Blocs actifs : {', '.join(self.active_blocks)}",
        ]
        for block, cols in self.block_columns.items():
            lines.append(f"  {block:<20} → {len(cols)} features")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AssemblyReport — rapport de la fusion
# ---------------------------------------------------------------------------

@dataclass
class AssemblyReport:
    """Rapport produit après chaque appel à assemble()."""
    n_accounts:      int = 0
    n_features:      int = 0
    active_blocks:   List[str] = field(default_factory=list)
    skipped_blocks:  List[str] = field(default_factory=list)
    n_nan_filled:    int = 0
    n_cols_dropped:  int = 0
    warnings:        List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"AssemblyReport : {self.n_accounts} comptes × {self.n_features} features",
            f"  Blocs actifs   : {', '.join(self.active_blocks)}",
            f"  Blocs ignorés  : {', '.join(self.skipped_blocks)}",
            f"  NaN comblés    : {self.n_nan_filled}",
            f"  Colonnes drop  : {self.n_cols_dropped}",
        ]
        for w in self.warnings:
            lines.append(f"  [!] {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FeatureAssembler — classe principale
# ---------------------------------------------------------------------------

class FeatureAssembler:
    """
    Assembleur central de features.

    Cycle de vie :
        1. add_block(name, df)  — enregistrer les blocs
        2. assemble()           — fusionner en matrice finale
        3. save_map(path)       — sauvegarder pour l'inférence

    Pour l'inférence :
        asm = FeatureAssembler.load_map(path)
        X = asm.transform(blocks_dict)
    """

    # Ordre canonique des blocs (garantit la reproductibilité)
    BLOCK_ORDER = [
        "tabular",
        "temporal",
        "text_basic",
        "text_embeddings",
        "structural",
        "relational",
    ]

    def __init__(self, account_id_col: str = AccountCols.ID) -> None:
        self.account_id_col = account_id_col
        self._blocks: Dict[str, pd.DataFrame] = {}
        self._feature_map: Optional[FeatureMap] = None

    # ------------------------------------------------------------------
    # Ajout de blocs
    # ------------------------------------------------------------------

    def add_block(self, name: str, df: pd.DataFrame) -> None:
        """
        Enregistre un bloc de features.

        Args:
            name : identifiant du bloc (ex: "tabular", "temporal")
            df   : DataFrame avec account_id + colonnes de features
        """
        if df is None or df.empty:
            logger.warning("Bloc '%s' vide — ignoré", name)
            return

        if self.account_id_col not in df.columns:
            raise ValueError(
                f"Le bloc '{name}' ne contient pas la colonne "
                f"'{self.account_id_col}'. Elle est obligatoire."
            )

        # Déduplique par account_id (garde la première occurrence)
        n_before = len(df)
        df = df.drop_duplicates(subset=[self.account_id_col]).reset_index(drop=True)
        if len(df) < n_before:
            logger.warning(
                "Bloc '%s' : %d doublons account_id supprimés",
                name, n_before - len(df)
            )

        self._blocks[name] = df
        logger.info("Bloc '%s' enregistré : %d comptes × %d features",
                    name, len(df), len(df.columns) - 1)

    def has_block(self, name: str) -> bool:
        return name in self._blocks

    def block_names(self) -> List[str]:
        return list(self._blocks.keys())

    # ------------------------------------------------------------------
    # Assemblage
    # ------------------------------------------------------------------

    def assemble(
        self,
        labels_df:         Optional[pd.DataFrame] = None,
        drop_constant_cols:  bool = True,
        drop_high_nan_cols:  float = 0.8,
        fill_value:          float = _DEFAULT_FILL,
        return_labels:       bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], FeatureMap, AssemblyReport]:
        """
        Fusionne tous les blocs enregistrés en une matrice finale.

        Args:
            labels_df          : DataFrame avec account_id + label (optionnel)
            drop_constant_cols : supprimer les colonnes avec une seule valeur
            drop_high_nan_cols : supprimer les colonnes avec > x% de NaN (0.0–1.0)
            fill_value         : valeur de remplissage pour les NaN restants
            return_labels      : aligner et retourner les labels

        Returns:
            (X, y, feature_map, report)
            X            : DataFrame de features finale (float)
            y            : Series des labels (ou None)
            feature_map  : FeatureMap décrivant la matrice
            report       : AssemblyReport
        """
        report = AssemblyReport()

        if not self._blocks:
            raise RuntimeError("Aucun bloc enregistré. Utiliser add_block() d'abord.")

        # ---- 1. Construire la liste des account_ids de référence ----
        # Prendre l'union de tous les account_ids connus
        all_ids: pd.Index = pd.Index([], name=self.account_id_col)
        for name in self._get_ordered_blocks():
            df = self._blocks[name]
            all_ids = all_ids.union(df[self.account_id_col])

        master = pd.DataFrame({self.account_id_col: all_ids})
        report.n_accounts = len(master)

        # ---- 2. Fusion des blocs par LEFT JOIN ----
        block_cols: Dict[str, List[str]] = {}
        ordered = self._get_ordered_blocks()

        for name in ordered:
            df = self._blocks[name]
            # Colonnes de features (exclure account_id)
            feat_cols = [c for c in df.columns if c != self.account_id_col]

            # Préfixe les colonnes pour éviter les collisions
            prefix = f"{name}__"
            rename_map = {
                c: f"{prefix}{c}" if not c.startswith(prefix) else c
                for c in feat_cols
            }
            df = df.rename(columns=rename_map)
            prefixed_cols = list(rename_map.values())

            master = master.merge(
                df[[self.account_id_col] + prefixed_cols],
                on=self.account_id_col,
                how="left",
            )
            block_cols[name] = prefixed_cols
            report.active_blocks.append(name)
            logger.info("Bloc '%s' fusionné : +%d colonnes", name, len(prefixed_cols))

        # ---- 3. Séparer account_id du reste ----
        feature_df = master.drop(columns=[self.account_id_col])
        id_series  = master[self.account_id_col]

        # ---- 4. Nettoyage des colonnes non-numériques ----
        non_numeric = feature_df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
        if non_numeric:
            # Tente de convertir les booléens
            for col in non_numeric:
                if feature_df[col].dtype == object:
                    try:
                        feature_df[col] = feature_df[col].astype(float)
                        non_numeric.remove(col)
                    except (ValueError, TypeError):
                        pass
            # Si encore non-numériques, les supprimer
            still_non_num = feature_df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
            if still_non_num:
                feature_df = feature_df.drop(columns=still_non_num)
                report.n_cols_dropped += len(still_non_num)
                report.warnings.append(
                    f"Colonnes non-numériques supprimées : {still_non_num}"
                )

        # ---- 5. Supprimer colonnes avec trop de NaN ----
        if drop_high_nan_cols > 0:
            nan_rates = feature_df.isna().mean()
            high_nan  = nan_rates[nan_rates > drop_high_nan_cols].index.tolist()
            if high_nan:
                feature_df = feature_df.drop(columns=high_nan)
                report.n_cols_dropped += len(high_nan)
                report.warnings.append(
                    f"{len(high_nan)} colonnes supprimées (>{drop_high_nan_cols*100:.0f}% NaN)"
                )

        # ---- 6. Supprimer colonnes constantes ----
        if drop_constant_cols:
            constant = [
                c for c in feature_df.columns
                if feature_df[c].nunique(dropna=False) <= 1
            ]
            if constant:
                feature_df = feature_df.drop(columns=constant)
                report.n_cols_dropped += len(constant)
                report.warnings.append(
                    f"{len(constant)} colonnes constantes supprimées"
                )

        # ---- 7. Remplir les NaN restants ----
        n_nan_before = int(feature_df.isna().sum().sum())
        feature_df   = feature_df.fillna(fill_value)
        report.n_nan_filled = n_nan_before

        # ---- 8. Construire le FeatureMap ----
        final_cols = list(feature_df.columns)
        fill_vals  = {c: fill_value for c in final_cols}

        # Reconstruire block_cols en filtrant les colonnes finales (après drops)
        final_block_cols: Dict[str, List[str]] = {}
        for block, cols in block_cols.items():
            kept = [c for c in cols if c in final_cols]
            if kept:
                final_block_cols[block] = kept

        feature_map = FeatureMap(
            feature_columns = final_cols,
            block_columns   = final_block_cols,
            fill_values     = fill_vals,
            active_blocks   = list(final_block_cols.keys()),
            n_features      = len(final_cols),
        )
        self._feature_map = feature_map
        report.n_features = feature_map.n_features

        logger.info("Assemblage terminé : %d comptes × %d features",
                    report.n_accounts, report.n_features)

        # ---- 9. Réintégrer account_id comme index ----
        feature_df = feature_df.set_index(id_series)

        # ---- 10. Aligner les labels ----
        y: Optional[pd.Series] = None
        if return_labels and labels_df is not None:
            y = self._align_labels(labels_df, id_series)

        return feature_df, y, feature_map, report

    # ------------------------------------------------------------------
    # Transformation (inférence)
    # ------------------------------------------------------------------

    def transform(
        self,
        blocks: Dict[str, pd.DataFrame],
        feature_map: Optional[FeatureMap] = None,
    ) -> pd.DataFrame:
        """
        Transforme de nouveaux blocs en utilisant le FeatureMap existant.
        Garantit la même matrice de features qu'à l'entraînement.

        Args:
            blocks      : dict {block_name → DataFrame}
            feature_map : FeatureMap de référence (utilise self._feature_map si None)

        Returns:
            DataFrame avec exactement les mêmes colonnes que lors de l'entraînement
        """
        fm = feature_map or self._feature_map
        if fm is None:
            raise RuntimeError("Pas de FeatureMap disponible. Lancer assemble() ou load_map() d'abord.")

        # Enregistrer les nouveaux blocs
        asm_tmp = FeatureAssembler(self.account_id_col)
        for name, df in blocks.items():
            asm_tmp.add_block(name, df)

        # Assembler sans filtrage (on veut exactement les colonnes du FeatureMap)
        X_new, _, _, _ = asm_tmp.assemble(
            labels_df        = None,
            drop_constant_cols  = False,
            drop_high_nan_cols  = 0.0,
            fill_value          = _DEFAULT_FILL,
            return_labels       = False,
        )

        # Aligner sur les colonnes du FeatureMap
        X_new = _align_to_feature_map(X_new, fm)
        return X_new

    # ------------------------------------------------------------------
    # Sauvegarde / chargement du FeatureMap
    # ------------------------------------------------------------------

    def save_map(self, path: str | Path) -> None:
        """Sauvegarde le FeatureMap courant en JSON."""
        if self._feature_map is None:
            raise RuntimeError("Appeler assemble() avant save_map().")
        self._feature_map.save(path)

    @classmethod
    def load_map(cls, path: str | Path) -> "FeatureAssembler":
        """Charge un FeatureMap depuis un JSON pour l'inférence."""
        asm = cls()
        asm._feature_map = FeatureMap.load(path)
        return asm

    @property
    def feature_map(self) -> Optional[FeatureMap]:
        return self._feature_map

    # ------------------------------------------------------------------
    # Utilitaires internes
    # ------------------------------------------------------------------

    def _get_ordered_blocks(self) -> List[str]:
        """Retourne les blocs dans l'ordre canonique, puis les blocs inconnus à la fin."""
        known   = [b for b in self.BLOCK_ORDER if b in self._blocks]
        unknown = [b for b in self._blocks if b not in self.BLOCK_ORDER]
        return known + unknown

    def _align_labels(
        self,
        labels_df: pd.DataFrame,
        id_series: pd.Series,
    ) -> Optional[pd.Series]:
        """Aligne les labels sur les account_ids de la matrice finale."""
        if LabelCols.LABEL not in labels_df.columns:
            return None

        label_col = self.account_id_col
        if label_col not in labels_df.columns:
            logger.warning("labels_df n'a pas de colonne account_id — labels non alignés")
            return None

        label_map = labels_df.set_index(label_col)[LabelCols.LABEL]
        y = id_series.map(label_map)
        n_missing = int(y.isna().sum())
        if n_missing > 0:
            logger.warning("%d comptes sans label dans labels_df", n_missing)
        return y.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utilitaire standalone : alignment sur FeatureMap
# ---------------------------------------------------------------------------

def _align_to_feature_map(X: pd.DataFrame, fm: FeatureMap) -> pd.DataFrame:
    """
    Aligne un DataFrame de features sur le FeatureMap de référence.
    - Ajoute les colonnes manquantes avec leurs fill_values
    - Supprime les colonnes inconnues
    - Réordonne selon feature_columns
    """
    result = X.copy()

    # Ajouter les colonnes manquantes
    for col in fm.feature_columns:
        if col not in result.columns:
            result[col] = fm.fill_values.get(col, _DEFAULT_FILL)

    # Réordonner + sélectionner seulement les colonnes du FeatureMap
    result = result[fm.feature_columns]
    return result


# ---------------------------------------------------------------------------
# Fonction de convenance pour usage rapide
# ---------------------------------------------------------------------------

def assemble_features(
    blocks:            Dict[str, pd.DataFrame],
    labels_df:         Optional[pd.DataFrame] = None,
    account_id_col:    str = AccountCols.ID,
    drop_constant_cols:  bool = True,
    drop_high_nan_cols:  float = 0.8,
    fill_value:          float = _DEFAULT_FILL,
    map_save_path:       Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], FeatureMap, AssemblyReport]:
    """
    Fonction de convenance : assemble des blocs en une seule étape.

    Args:
        blocks             : dict {block_name → DataFrame}
        labels_df          : DataFrame avec account_id + label
        account_id_col     : nom de la colonne clé
        drop_constant_cols : supprimer les colonnes constantes
        drop_high_nan_cols : seuil de NaN pour suppression (ex: 0.8 = 80%)
        fill_value         : valeur de remplissage pour les NaN
        map_save_path      : chemin JSON pour sauvegarder le FeatureMap

    Returns:
        (X, y, feature_map, report)
    """
    asm = FeatureAssembler(account_id_col)

    for name, df in blocks.items():
        if df is not None and not df.empty:
            asm.add_block(name, df)

    X, y, fm, report = asm.assemble(
        labels_df          = labels_df,
        drop_constant_cols   = drop_constant_cols,
        drop_high_nan_cols   = drop_high_nan_cols,
        fill_value           = fill_value,
        return_labels        = labels_df is not None,
    )

    if map_save_path:
        asm.save_map(map_save_path)

    return X, y, fm, report
