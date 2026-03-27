# -*- coding: utf-8 -*-
"""
src/inference/predict.py
-------------------------
Module d'inférence : prédiction de probabilités et labels pour de nouvelles données.

Responsabilités :
    - Charger un modèle sauvegardé (ou accepter un modèle en mémoire)
    - Aligner les features du jeu test sur le FeatureMap d'entraînement
    - Retourner probabilités + labels + métadonnées d'incertitude

Philosophie anti-faux-positifs :
    Le seuil par défaut est CONSERVATEUR (favorise le rappel sur la précision).
    "Le doute profite à l'humain."

Usage :
    from src.inference.predict import Predictor

    predictor = Predictor.from_model(model, feature_map)
    predictions = predictor.predict(X_test, account_ids=ids)
    print(predictions.head())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.models._base import BotDetectorBase
from src.features.assembler import FeatureMap, _align_to_feature_map

logger = logging.getLogger(__name__)

# Seuil conservateur par défaut (prudent sur les faux positifs)
_DEFAULT_THRESHOLD = 0.5

# Labels textuels
_LABEL_BOT   = "bot"
_LABEL_HUMAN = "human"

# Zones de confiance
_HIGH_CONF_BOT   = 0.80   # probabilité >= 80% → bot très probable
_HIGH_CONF_HUMAN = 0.20   # probabilité <= 20% → humain très probable


# ---------------------------------------------------------------------------
# PredictionResult — conteneur de sortie
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Résultats d'une inférence.

    Colonnes du DataFrame `df` :
        account_id   : identifiant du compte
        prob_bot     : probabilité d'être un bot (0.0–1.0)
        label        : prédiction binaire (0/1)
        label_text   : "bot" ou "human"
        confidence   : "high" | "medium" | "low"
        is_uncertain : True si probabilité proche du seuil (dans ±0.15)
    """
    df: pd.DataFrame
    threshold: float
    model_name: str
    n_accounts:  int
    n_predicted_bots: int
    n_uncertain: int

    def summary(self) -> str:
        pct_bot = 100 * self.n_predicted_bots / max(self.n_accounts, 1)
        pct_unc = 100 * self.n_uncertain / max(self.n_accounts, 1)
        return (
            f"[{self.model_name}] {self.n_accounts} comptes → "
            f"{self.n_predicted_bots} bots ({pct_bot:.1f}%)  "
            f"| {self.n_uncertain} incertains ({pct_unc:.1f}%)  "
            f"| seuil={self.threshold:.2f}"
        )


# ---------------------------------------------------------------------------
# Predictor — classe principale
# ---------------------------------------------------------------------------

class Predictor:
    """
    Moteur d'inférence.

    Peut être instancié depuis :
        - Un modèle en mémoire : Predictor.from_model(model, feature_map)
        - Un modèle sauvegardé : Predictor.from_path(model_path, map_path, model_cls)
    """

    def __init__(
        self,
        model:       BotDetectorBase,
        feature_map: Optional[FeatureMap] = None,
        threshold:   Optional[float] = None,
    ) -> None:
        self.model       = model
        self.feature_map = feature_map
        self.threshold   = threshold or model.threshold or _DEFAULT_THRESHOLD

    # ------------------------------------------------------------------
    # Constructeurs alternatifs
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model:       BotDetectorBase,
        feature_map: Optional[FeatureMap] = None,
        threshold:   Optional[float] = None,
    ) -> "Predictor":
        """Instancie depuis un modèle déjà en mémoire."""
        return cls(model=model, feature_map=feature_map, threshold=threshold)

    @classmethod
    def from_path(
        cls,
        model_path:  Union[str, Path],
        map_path:    Optional[Union[str, Path]] = None,
        model_cls:   Optional[type] = None,
        threshold:   Optional[float] = None,
    ) -> "Predictor":
        """
        Charge un modèle depuis le disque.

        Args:
            model_path : chemin vers le fichier .joblib (sans extension)
            map_path   : chemin vers le FeatureMap JSON (optionnel)
            model_cls  : classe du modèle pour load() (ex: LightGBMDetector)
            threshold  : seuil à utiliser (lit le .json si None)
        """
        if model_cls is None:
            from src.models.baseline_lr import LogisticRegressionDetector
            model_cls = LogisticRegressionDetector

        model = model_cls.load(model_path)
        feature_map = FeatureMap.load(map_path) if map_path else None
        thr = threshold or model.threshold

        logger.info("Modèle chargé : %s (thr=%.2f)", model.name, thr)
        return cls(model=model, feature_map=feature_map, threshold=thr)

    # ------------------------------------------------------------------
    # Prédiction principale
    # ------------------------------------------------------------------

    def predict(
        self,
        X:           pd.DataFrame,
        account_ids: Optional[pd.Series] = None,
        threshold:   Optional[float] = None,
        uncertainty_margin: float = 0.15,
    ) -> PredictionResult:
        """
        Prédit les probabilités et labels pour chaque compte.

        Args:
            X                  : matrice de features (alignée ou non sur le FeatureMap)
            account_ids        : identifiants de compte (index du DataFrame si None)
            threshold          : seuil de décision (utilise self.threshold si None)
            uncertainty_margin : marge autour du seuil pour marquer "uncertain"

        Returns:
            PredictionResult avec le DataFrame de prédictions
        """
        thr = threshold or self.threshold

        # Aligner sur le FeatureMap si disponible
        if self.feature_map is not None:
            X = _align_to_feature_map(X, self.feature_map)
            logger.debug("Features alignées sur FeatureMap (%d colonnes)", len(X.columns))

        # Probabilités
        prob = self._safe_predict_proba(X)

        # Labels binaires
        labels = (prob >= thr).astype(int)

        # IDs
        if account_ids is not None:
            ids = account_ids.reset_index(drop=True)
        elif X.index.name or not X.index.equals(pd.RangeIndex(len(X))):
            ids = pd.Series(X.index, name="account_id")
        else:
            ids = pd.Series(range(len(X)), name="account_id")

        # Confiance
        confidence = self._confidence_level(prob, thr)
        is_uncertain = np.abs(prob - thr) <= uncertainty_margin

        # Assemblage du DataFrame de sortie
        df_out = pd.DataFrame({
            "account_id":  ids.values,
            "prob_bot":    np.round(prob, 4),
            "label":       labels,
            "label_text":  np.where(labels == 1, _LABEL_BOT, _LABEL_HUMAN),
            "confidence":  confidence,
            "is_uncertain": is_uncertain.astype(bool),
        })

        n_bots = int(labels.sum())
        n_unc  = int(is_uncertain.sum())

        result = PredictionResult(
            df               = df_out,
            threshold        = thr,
            model_name       = self.model.name,
            n_accounts       = len(df_out),
            n_predicted_bots = n_bots,
            n_uncertain      = n_unc,
        )
        logger.info(result.summary())
        return result

    def predict_proba_only(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne seulement le vecteur de probabilités (plus rapide)."""
        if self.feature_map is not None:
            X = _align_to_feature_map(X, self.feature_map)
        return self._safe_predict_proba(X)

    # ------------------------------------------------------------------
    # Utilitaires internes
    # ------------------------------------------------------------------

    def _safe_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Appel sécurisé à predict_proba avec gestion des erreurs."""
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error("Erreur predict_proba : %s", e)
            raise

    @staticmethod
    def _confidence_level(prob: np.ndarray, thr: float) -> np.ndarray:
        """Catégorise la confiance de chaque prédiction."""
        confidence = np.full(len(prob), "medium", dtype=object)
        confidence[(prob >= _HIGH_CONF_BOT) | (prob <= _HIGH_CONF_HUMAN)] = "high"
        confidence[
            (prob > (_HIGH_CONF_HUMAN + 0.10)) &
            (prob < (_HIGH_CONF_BOT  - 0.10))
        ] = "low"
        return confidence
