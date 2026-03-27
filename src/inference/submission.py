# -*- coding: utf-8 -*-
"""
src/inference/submission.py
----------------------------
Préparation et export de la soumission finale pour la compétition.

Responsabilités :
    - Formater les prédictions dans le format attendu par la compétition
    - Supporter plusieurs formats de sortie (CSV, JSON, JSONL)
    - Ajouter métadonnées de traçabilité (modèle, seuil, timestamp)
    - Gérer l'ensemble de modèles (prédictions agrégées multi-modèles)
    - Valider la soumission avant export (colonnes, types, plages)

Format de soumission standard (configurable le jour J) :
    account_id, label         → classification binaire
    account_id, prob_bot      → probabilités
    account_id, label, score  → les deux

Usage :
    from src.inference.submission import SubmissionBuilder

    builder = SubmissionBuilder(format="default")
    df_sub  = builder.from_prediction(prediction_result)
    builder.save(df_sub, "submissions/sub_v1.csv")

    # Ou : ensembling de plusieurs modèles
    df_sub = builder.from_ensemble([result_lgb, result_xgb], method="mean")
    builder.save(df_sub, "submissions/ensemble_v1.csv")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from src.inference.predict import PredictionResult

logger = logging.getLogger(__name__)

# Formats de soumission supportés
SubmissionFormat = Literal["default", "proba_only", "full", "custom"]


# ---------------------------------------------------------------------------
# Formats prédéfinis
# ---------------------------------------------------------------------------

# Format "default" : id + label binaire (le plus courant en compétition)
_DEFAULT_COLS = {"id_col": "account_id", "label_col": "label"}

# Format "proba" : id + probabilité (pour compétitions évaluées sur AUROC)
_PROBA_COLS   = {"id_col": "account_id", "proba_col": "prob_bot"}

# Format "full" : tout inclus
_FULL_COLS    = {"id_col": "account_id", "label_col": "label", "proba_col": "prob_bot"}


# ---------------------------------------------------------------------------
# SubmissionBuilder
# ---------------------------------------------------------------------------

class SubmissionBuilder:
    """
    Constructeur de soumission configurable.

    Args:
        format       : "default" (id+label) | "proba_only" | "full" | "custom"
        id_col       : nom de la colonne ID dans la sortie (ex: "id", "user_id")
        label_col    : nom de la colonne label dans la sortie
        proba_col    : nom de la colonne probabilité dans la sortie
        add_metadata : ajouter un fichier .json de métadonnées à côté
    """

    def __init__(
        self,
        format:       SubmissionFormat = "default",
        id_col:       str  = "account_id",
        label_col:    str  = "label",
        proba_col:    str  = "prob_bot",
        add_metadata: bool = True,
    ) -> None:
        self.format       = format
        self.id_col       = id_col
        self.label_col    = label_col
        self.proba_col    = proba_col
        self.add_metadata = add_metadata

    # ------------------------------------------------------------------
    # Construction depuis une PredictionResult
    # ------------------------------------------------------------------

    def from_prediction(
        self,
        result:     PredictionResult,
        sort_by_id: bool = True,
    ) -> pd.DataFrame:
        """
        Formate une PredictionResult en DataFrame de soumission.

        Args:
            result     : PredictionResult produit par Predictor.predict()
            sort_by_id : trier par account_id

        Returns:
            DataFrame prêt à exporter
        """
        df = result.df.copy()
        return self._format(df, sort_by_id=sort_by_id)

    def from_probabilities(
        self,
        account_ids: pd.Series,
        probabilities: np.ndarray,
        threshold:  float = 0.5,
        sort_by_id: bool = True,
    ) -> pd.DataFrame:
        """
        Formate directement depuis des probabilités brutes.

        Args:
            account_ids   : Series d'identifiants
            probabilities : array de probabilités (0.0–1.0)
            threshold     : seuil pour les labels binaires
        """
        df = pd.DataFrame({
            "account_id": account_ids.values,
            "prob_bot":   np.round(probabilities, 4),
            "label":      (probabilities >= threshold).astype(int),
        })
        return self._format(df, sort_by_id=sort_by_id)

    # ------------------------------------------------------------------
    # Ensembling multi-modèles
    # ------------------------------------------------------------------

    def from_ensemble(
        self,
        results:    List[PredictionResult],
        method:     Literal["mean", "max", "vote", "weighted"] = "mean",
        weights:    Optional[List[float]] = None,
        threshold:  float = 0.5,
        sort_by_id: bool = True,
    ) -> pd.DataFrame:
        """
        Combine les prédictions de plusieurs modèles.

        Méthodes :
            "mean"     → moyenne des probabilités (soft voting)
            "max"      → probabilité max parmi les modèles (agressif)
            "vote"     → vote majoritaire sur les labels binaires (hard)
            "weighted" → moyenne pondérée (weights doit être fourni)

        Args:
            results   : liste de PredictionResult
            method    : méthode d'agrégation
            weights   : poids par modèle (normalized auto si fourni)
            threshold : seuil pour les labels finaux

        Returns:
            DataFrame de soumission
        """
        if not results:
            raise ValueError("La liste de résultats est vide")

        # Aligner toutes les probabilités sur les mêmes account_ids
        ref_ids = results[0].df["account_id"].reset_index(drop=True)
        proba_matrix = []

        for result in results:
            df_r = result.df.set_index("account_id")["prob_bot"]
            # Réindexer sur les IDs de référence (NaN si absent)
            aligned = df_r.reindex(ref_ids).fillna(0.5).values
            proba_matrix.append(aligned)

        proba_matrix = np.array(proba_matrix)  # shape: (n_models, n_accounts)

        # Agrégation
        if method == "mean":
            final_prob = proba_matrix.mean(axis=0)

        elif method == "max":
            final_prob = proba_matrix.max(axis=0)

        elif method == "weighted":
            if weights is None:
                raise ValueError("'weights' doit être fourni pour method='weighted'")
            w = np.array(weights, dtype=float)
            w /= w.sum()   # normalisation
            final_prob = (proba_matrix * w[:, None]).sum(axis=0)

        elif method == "vote":
            # Vote majoritaire sur les labels
            labels_matrix = (proba_matrix >= threshold).astype(int)
            final_labels  = (labels_matrix.mean(axis=0) >= 0.5).astype(int)
            final_prob    = final_labels.astype(float)   # 0 ou 1 dans ce mode

        else:
            raise ValueError(f"Méthode d'ensemble inconnue : '{method}'")

        model_names = " + ".join(r.model_name for r in results)
        logger.info("Ensemble [%s] via %s : %d comptes", model_names, method, len(ref_ids))

        df = pd.DataFrame({
            "account_id": ref_ids.values,
            "prob_bot":   np.round(final_prob, 4),
            "label":      (final_prob >= threshold).astype(int),
        })
        return self._format(df, sort_by_id=sort_by_id)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(
        self,
        df:           pd.DataFrame,
        path:         Union[str, Path],
        file_format:  Literal["csv", "json", "jsonl"] = "csv",
        metadata:     Optional[Dict] = None,
    ) -> Path:
        """
        Sauvegarde la soumission sur le disque.

        Args:
            df          : DataFrame de soumission
            path        : chemin de sortie (sans extension si auto)
            file_format : "csv" | "json" | "jsonl"
            metadata    : dict de métadonnées additionnelles

        Returns:
            Path du fichier créé
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Valider avant export
        self._validate(df)

        # Export principal
        if file_format == "csv":
            out_path = path.with_suffix(".csv")
            df.to_csv(out_path, index=False)

        elif file_format == "json":
            out_path = path.with_suffix(".json")
            df.to_json(out_path, orient="records", indent=2, force_ascii=False)

        elif file_format == "jsonl":
            out_path = path.with_suffix(".jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    f.write(row.to_json(force_ascii=False) + "\n")

        else:
            raise ValueError(f"Format inconnu : '{file_format}'")

        logger.info("Soumission exportée : %s (%d lignes)", out_path, len(df))

        # Métadonnées
        if self.add_metadata:
            meta = {
                "timestamp":   datetime.utcnow().isoformat() + "Z",
                "n_accounts":  len(df),
                "format":      self.format,
                "file_format": file_format,
                "columns":     list(df.columns),
            }
            if metadata:
                meta.update(metadata)
            meta_path = path.with_suffix(".meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Métadonnées : %s", meta_path)

        return out_path

    # ------------------------------------------------------------------
    # Formatage interne
    # ------------------------------------------------------------------

    def _format(self, df: pd.DataFrame, sort_by_id: bool = True) -> pd.DataFrame:
        """Sélectionne et renomme les colonnes selon le format choisi."""
        # Renommer account_id → id_col si différent
        if self.id_col != "account_id" and "account_id" in df.columns:
            df = df.rename(columns={"account_id": self.id_col})

        if self.format == "default":
            cols = [self.id_col, "label"]
            # Renommer label si nécessaire
            if self.label_col != "label" and "label" in df.columns:
                df = df.rename(columns={"label": self.label_col})
                cols = [self.id_col, self.label_col]

        elif self.format == "proba_only":
            if self.proba_col != "prob_bot" and "prob_bot" in df.columns:
                df = df.rename(columns={"prob_bot": self.proba_col})
            cols = [self.id_col, self.proba_col]

        elif self.format == "full":
            cols = [c for c in [self.id_col, "label", "prob_bot",
                                 "label_text", "confidence", "is_uncertain"]
                    if c in df.columns]

        else:   # custom ou fallback
            cols = [c for c in df.columns if c in (
                self.id_col, "account_id", "label", "prob_bot",
                self.label_col, self.proba_col
            )]

        # Garder seulement les colonnes disponibles
        final_cols = [c for c in cols if c in df.columns]
        df = df[final_cols]

        if sort_by_id and self.id_col in df.columns:
            df = df.sort_values(self.id_col).reset_index(drop=True)

        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Valide la soumission avant export."""
        if df.empty:
            raise ValueError("La soumission est vide")

        if self.id_col not in df.columns:
            raise ValueError(f"Colonne ID manquante : '{self.id_col}'")

        # Vérifier absence de doublons d'IDs
        n_dup = df.duplicated(subset=[self.id_col]).sum()
        if n_dup > 0:
            logger.warning("%d doublons d'account_id dans la soumission !", n_dup)

        # Vérifier plage des probabilités si présentes
        prob_col = next((c for c in df.columns if "prob" in c.lower()), None)
        if prob_col:
            proba = df[prob_col].dropna()
            if (proba < 0).any() or (proba > 1).any():
                raise ValueError("Probabilités hors de [0, 1] détectées")

        # Vérifier les labels binaires si présents
        lbl_col = next((c for c in df.columns if "label" in c.lower()
                        and "text" not in c.lower()), None)
        if lbl_col and df[lbl_col].dtype in (int, float, "int64", "float64"):
            invalid = ~df[lbl_col].isin([0, 1])
            if invalid.any():
                logger.warning("%d labels non binaires dans la soumission", invalid.sum())

        logger.info("Validation OK : %d lignes, colonnes=%s",
                    len(df), list(df.columns))
