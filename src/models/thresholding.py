# -*- coding: utf-8 -*-
"""
src/models/thresholding.py
---------------------------
Logique de seuil et de décision prudente pour BotOrNot.

Philosophie :
    "En cas de doute, le compte est humain."
    Un faux positif (humain classé bot) est plus coûteux qu'un faux négatif
    dans la plupart des contextes réels. Ce module rend ce choix explicite
    et configurable.

Fonctionnalités :
    - Optimisation du seuil selon plusieurs métriques (F1, precision, recall,
      balanced_accuracy, ou seuil conservateur fixe)
    - Zone de refus / incertitude : les comptes dans ±margin → label "human"
    - Résolution de conflits entre modules (si plusieurs signaux contradictoires)
    - Mode "strict" (haute précision) vs "recall" (attrape le max de bots)

Usage :
    from src.models.thresholding import ThresholdOptimizer, DecisionEngine

    # Optimisation
    opt = ThresholdOptimizer(metric="f1")
    thr = opt.find(y_val, y_prob_val)
    print(f"Seuil optimal F1 : {thr}")

    # Décision prudente
    engine = DecisionEngine(threshold=thr, uncertainty_margin=0.12)
    decisions = engine.decide(y_prob_test, account_ids=ids)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Modes de seuil prédéfinis
ThresholdMetric = Literal["f1", "precision", "recall", "balanced", "conservative", "custom"]

# Seuils conservateurs par mode
_CONSERVATIVE_THRESHOLD = 0.65   # haute précision : peu de faux positifs
_AGGRESSIVE_THRESHOLD   = 0.35   # haut rappel : attrape plus de bots


# ---------------------------------------------------------------------------
# ThresholdOptimizer
# ---------------------------------------------------------------------------

class ThresholdOptimizer:
    """
    Recherche le seuil optimal sur un jeu de validation.

    Métriques supportées :
        "f1"          → maximise le F1-score (équilibre)
        "precision"   → maximise la précision (peu de faux positifs)
        "recall"      → maximise le rappel (peu de faux négatifs)
        "balanced"    → maximise le G-mean (TPR × TNR)
        "conservative"→ seuil fixe conservateur (0.65)
        "custom"      → utilise le seuil passé directement
    """

    def __init__(
        self,
        metric:    ThresholdMetric = "f1",
        n_steps:   int = 90,
        thr_min:   float = 0.05,
        thr_max:   float = 0.95,
    ) -> None:
        self.metric  = metric
        self.n_steps = n_steps
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.best_threshold_: Optional[float] = None
        self.best_score_:     Optional[float] = None
        self.threshold_curve_: Optional[pd.DataFrame] = None

    def find(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        custom_threshold: Optional[float] = None,
    ) -> float:
        """
        Trouve et retourne le seuil optimal.

        Args:
            y_true           : labels vrais (0/1)
            y_prob           : probabilités prédites
            custom_threshold : seuil fixe (utilisé si metric="custom")

        Returns:
            Seuil optimal (float)
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        if self.metric == "conservative":
            self.best_threshold_ = _CONSERVATIVE_THRESHOLD
            self.best_score_     = float("nan")
            logger.info("[Threshold] Mode conservateur → %.2f", self.best_threshold_)
            return self.best_threshold_

        if self.metric == "custom":
            if custom_threshold is None:
                raise ValueError("custom_threshold doit être fourni si metric='custom'")
            self.best_threshold_ = float(custom_threshold)
            self.best_score_     = float("nan")
            return self.best_threshold_

        thresholds  = np.linspace(self.thr_min, self.thr_max, self.n_steps)
        best        = -1.0
        best_thr    = 0.5
        curve_rows  = []

        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            score  = self._score(y_true, y_pred, thr, y_prob)
            curve_rows.append({"threshold": round(float(thr), 4), "score": round(score, 4)})
            if score > best:
                best     = score
                best_thr = float(thr)

        self.best_threshold_  = round(best_thr, 4)
        self.best_score_      = round(best, 4)
        self.threshold_curve_ = pd.DataFrame(curve_rows)

        logger.info("[Threshold] Métrique=%s → optimal=%.4f (score=%.4f)",
                    self.metric, self.best_threshold_, self.best_score_)
        return self.best_threshold_

    def _score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thr:    float,
        y_prob: np.ndarray,
    ) -> float:
        """Calcule la métrique pour un seuil donné."""
        from sklearn.metrics import f1_score, precision_score, recall_score

        tp  = float(((y_pred == 1) & (y_true == 1)).sum())
        tn  = float(((y_pred == 0) & (y_true == 0)).sum())
        n_p = float(y_true.sum())
        n_n = float((y_true == 0).sum())

        if self.metric == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))

        elif self.metric == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0))

        elif self.metric == "recall":
            return float(recall_score(y_true, y_pred, zero_division=0))

        elif self.metric == "balanced":
            tpr = tp / max(n_p, 1)
            tnr = tn / max(n_n, 1)
            return float((tpr * tnr) ** 0.5)   # G-mean

        return 0.0

    def curve(self) -> Optional[pd.DataFrame]:
        """Retourne la courbe score vs seuil (si calculée)."""
        return self.threshold_curve_


# ---------------------------------------------------------------------------
# Résolution de conflits
# ---------------------------------------------------------------------------

@dataclass
class ModuleSignal:
    """Signal d'un module de détection."""
    module_name: str
    prob_bot:    float   # 0.0–1.0
    weight:      float = 1.0     # importance relative du module
    available:   bool  = True    # signal présent dans ce dataset


def resolve_conflict(
    signals:   List[ModuleSignal],
    method:    Literal["weighted_mean", "max", "vote"] = "weighted_mean",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Agrège plusieurs signaux de modules potentiellement contradictoires.

    Philosophie :
        - "weighted_mean" : pondère selon la fiabilité du module
        - "max"           : conservateur du côté de la sécurité (plus agressif)
        - "vote"          : majorité simple

    Args:
        signals   : liste de ModuleSignal
        method    : méthode d'agrégation
        threshold : seuil de décision finale

    Returns:
        dict {
            "final_prob"  : probabilité agrégée,
            "final_label" : 0 ou 1,
            "n_modules"   : nb de modules actifs,
            "agreement"   : fraction de modules qui votent bot,
            "conflict"    : True si les modules sont très divisés
        }
    """
    active = [s for s in signals if s.available]
    if not active:
        logger.warning("Aucun signal disponible → défaut humain (0)")
        return {"final_prob": 0.0, "final_label": 0, "n_modules": 0,
                "agreement": 0.0, "conflict": False}

    probs   = np.array([s.prob_bot for s in active])
    weights = np.array([s.weight   for s in active], dtype=float)
    weights /= weights.sum()   # normalisation

    if method == "weighted_mean":
        final_prob = float(np.dot(probs, weights))

    elif method == "max":
        final_prob = float(probs.max())

    elif method == "vote":
        votes      = (probs >= threshold).astype(int)
        final_prob = float(votes.mean())   # fraction de modules qui votent bot

    else:
        raise ValueError(f"Méthode inconnue : '{method}'")

    agreement = float((probs >= threshold).mean())
    conflict   = (agreement > 0.2) and (agreement < 0.8)   # modules divisés

    return {
        "final_prob":  round(final_prob, 4),
        "final_label": int(final_prob >= threshold),
        "n_modules":   len(active),
        "agreement":   round(agreement, 4),
        "conflict":    conflict,
    }


# ---------------------------------------------------------------------------
# DecisionEngine — moteur de décision prudente
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Moteur de décision prudente avec zone d'incertitude.

    Règles appliquées dans l'ordre :
        1. prob_bot < low_bound   → HUMAN (sûr)
        2. prob_bot > high_bound  → BOT   (sûr)
        3. Entre les deux         → UNCERTAIN → défaut = HUMAN (prudent)
        4. Si conflit entre modules → UNCERTAIN → défaut = HUMAN

    Args:
        threshold          : seuil central de décision
        uncertainty_margin : demi-largeur de la zone d'incertitude
        default_uncertain  : label par défaut si uncertain (0=human, 1=bot)
        require_agreement  : fraction minimale de modules qui doivent s'accorder
    """

    def __init__(
        self,
        threshold:           float = 0.5,
        uncertainty_margin:  float = 0.12,
        default_uncertain:   int   = 0,       # 0 = humain par défaut
        require_agreement:   float = 0.0,     # 0 = pas d'exigence de consensus
    ) -> None:
        self.threshold          = threshold
        self.uncertainty_margin = uncertainty_margin
        self.default_uncertain  = default_uncertain
        self.require_agreement  = require_agreement

        self.low_bound  = max(0.0, threshold - uncertainty_margin)
        self.high_bound = min(1.0, threshold + uncertainty_margin)

        logger.info(
            "[DecisionEngine] Zone sûre humain : <%.2f | Zone sûre bot : >%.2f | "
            "Incertitude : [%.2f, %.2f] → défaut=%s",
            self.low_bound, self.high_bound,
            self.low_bound, self.high_bound,
            "HUMAN" if default_uncertain == 0 else "BOT",
        )

    def decide_one(
        self,
        prob_bot:   float,
        agreement:  Optional[float] = None,
        conflict:   bool = False,
    ) -> Dict[str, object]:
        """
        Prend une décision pour un compte.

        Args:
            prob_bot  : probabilité brute (ou agrégée) d'être un bot
            agreement : fraction de modules en accord
            conflict  : True si les modules sont divisés

        Returns:
            dict {label, label_text, confidence, is_uncertain, reason}
        """
        is_uncertain = False
        reason       = "standard"

        # Conflit entre modules → incertitude forcée
        if conflict or (agreement is not None and agreement < self.require_agreement):
            is_uncertain = True
            reason = "module_conflict"

        # Zone d'incertitude → incertitude
        elif self.low_bound <= prob_bot <= self.high_bound:
            is_uncertain = True
            reason = "near_threshold"

        # Décision
        if is_uncertain:
            label = self.default_uncertain
        elif prob_bot < self.low_bound:
            label = 0
        else:
            label = 1

        # Confiance
        distance = abs(prob_bot - self.threshold)
        if is_uncertain:
            confidence = "low"
        elif distance > self.uncertainty_margin * 2:
            confidence = "high"
        else:
            confidence = "medium"

        return {
            "label":        label,
            "label_text":   "bot" if label == 1 else "human",
            "confidence":   confidence,
            "is_uncertain": is_uncertain,
            "reason":       reason,
        }

    def decide(
        self,
        y_prob:      np.ndarray,
        account_ids: Optional[pd.Series] = None,
        agreements:  Optional[np.ndarray] = None,
        conflicts:   Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Applique la décision prudente à un vecteur de probabilités.

        Args:
            y_prob       : probabilités (0.0–1.0)
            account_ids  : identifiants de compte
            agreements   : fraction d'accord par compte (optionnel)
            conflicts    : flags de conflit par compte (optionnel)

        Returns:
            DataFrame avec account_id, prob_bot, label, label_text,
                       confidence, is_uncertain, reason
        """
        n = len(y_prob)
        rows = []
        for i in range(n):
            prob = float(y_prob[i])
            agr  = float(agreements[i]) if agreements is not None else None
            cfl  = bool(conflicts[i])   if conflicts  is not None else False

            dec = self.decide_one(prob, agreement=agr, conflict=cfl)
            rows.append({
                "account_id":  account_ids.iloc[i] if account_ids is not None else i,
                "prob_bot":    round(prob, 4),
                **dec,
            })

        df = pd.DataFrame(rows)

        n_bots = int((df["label"] == 1).sum())
        n_unc  = int(df["is_uncertain"].sum())
        logger.info(
            "[DecisionEngine] %d comptes → %d bots (%.1f%%) | %d incertains (%.1f%%)",
            n, n_bots, 100*n_bots/max(n,1), n_unc, 100*n_unc/max(n,1),
        )
        return df

    def summary_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Retourne des statistiques de décision sur un DataFrame de résultats."""
        n = len(df)
        return {
            "n_total":          n,
            "n_bot":            int((df["label"] == 1).sum()),
            "n_human":          int((df["label"] == 0).sum()),
            "n_uncertain":      int(df["is_uncertain"].sum()),
            "pct_bot":          round(100 * (df["label"] == 1).mean(), 2),
            "pct_uncertain":    round(100 * df["is_uncertain"].mean(), 2),
            "pct_high_conf":    round(100 * (df["confidence"] == "high").mean(), 2),
        }
