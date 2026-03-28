# -*- coding: utf-8 -*-
"""
src/inference/consensus.py
===========================
Scoring de consensus inter-modules — V1 et V2

V1 (stable, toujours actif) :
  Moyenne pondérée des probabilités modulaires.

V2 (flag: enabled=False par defaut — RULES.md §2) :
  En plus de la moyenne ponderee :
  - n_agree        : nb de modules signalant "bot"
  - mean_confidence: confiance moyenne (distance a 0.5)
  - max_spread     : ecart max entre les modules
  - structural_boost: bonus si signal structurel fort
  → consensus_score: score composite explicite [0, 1]

Usage :
    from src.inference.consensus import ConsensusScorer, ConsensusScorerV2

    v1 = ConsensusScorer()
    proba = v1.score(block_probas)                      # -> np.ndarray

    v2 = ConsensusScorerV2(enabled=False)               # desactive par defaut
    df = v2.score(block_probas)                         # -> pd.DataFrame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Poids par défaut
# ─────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "tabular":    0.35,
    "temporal":   0.30,
    "text_basic": 0.15,
    "text_model": 0.10,
    "structural": 0.10,
}

STRUCTURAL_MODULES = {"structural", "temporal"}


# ─────────────────────────────────────────────────────────────
# V1 — Moyenne pondérée (stable, pas de flag requis)
# ─────────────────────────────────────────────────────────────

@dataclass
class ConsensusScorerConfig:
    """Configuration du scorer V1."""
    weights:         Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    fallback_weight: float            = 1.0
    bot_threshold:   float            = 0.50


class ConsensusScorer:
    """
    V1 — Scoring par moyenne pondérée simple.
    Comportement stable, déterministe, par défaut dans le pipeline.
    """

    def __init__(self, config: Optional[ConsensusScorerConfig] = None):
        self.cfg = config or ConsensusScorerConfig()

    @classmethod
    def from_config(cls, cfg: dict) -> "ConsensusScorer":
        return cls(ConsensusScorerConfig(
            weights         = cfg.get("weights", dict(DEFAULT_WEIGHTS)),
            fallback_weight = cfg.get("fallback_weight", 1.0),
            bot_threshold   = cfg.get("bot_threshold", 0.50),
        ))

    def score(self, block_probas: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Retourne la probabilité finale (pondérée par module).

        Args:
            block_probas : {module_name: proba_array(n,)}

        Returns:
            np.ndarray shape (n,), valeurs dans [0, 1]
        """
        arrays = {k: v for k, v in block_probas.items()
                  if isinstance(v, np.ndarray) and len(v) > 0}
        if not arrays:
            return np.array([0.5])
        n = len(next(iter(arrays.values())))
        w_sum, w_tot = np.zeros(n), 0.0
        for name, arr in arrays.items():
            w = self.cfg.weights.get(name, self.cfg.fallback_weight)
            w_sum += arr * w
            w_tot += w
        return np.clip(w_sum / max(w_tot, 1e-9), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# V2 — Scoring enrichi (DISABLED par défaut)
# ─────────────────────────────────────────────────────────────

@dataclass
class ConsensusScorerV2Config:
    """
    Configuration du scorer V2.
    enabled=False par defaut (RULES.md §2 : flag obligatoire).

    La V2 ne doit etre activee qu'apres validation par benchmark comparatif
    (RULES.md §3 : gain AUROC >= 0.01 OU reduction FP, sigma <= 0.02).
    """
    enabled:                bool             = False   # ← NE PAS CHANGER SANS BENCHMARK
    weights:                Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    fallback_weight:        float            = 1.0
    bot_threshold:          float            = 0.50
    # Composantes du consensus_score
    agreement_weight:       float            = 0.25   # poids n_agree
    confidence_weight:      float            = 0.25   # poids mean_confidence
    spread_penalty_weight:  float            = 0.20   # poids max_spread (penalite)
    spread_alarm_threshold: float            = 0.40   # seuil conflit
    structural_boost:       float            = 0.05   # bonus structurel (additionnel)
    structural_threshold:   float            = 0.65   # seuil signal fort
    # Interpolation V1 vs consensus_score
    v2_alpha:               float            = 0.50   # 0 = pure consensus, 1 = pure V1


class ConsensusScorerV2:
    """
    V2 — Scoring de consensus enrichi avec métriques explicites.

    FLAG : enabled=False par défaut.
    Activer seulement si :
      - Gain AUROC >= 0.01 OU reduction FP nette
      - sigma(AUROC) <= 0.02
      - Validé par consensus_v2_compare.py

    Métriques supplémentaires :
      n_agree          — nb de modules signalant bot (>= bot_threshold)
      mean_confidence  — confiance moyenne (|p - 0.5| × 2) en [0, 1]
      max_spread       — écart max entre modules (0 = unanime, 1 = max conflit)
      structural_boost — bonus additionnel si module structurel signal fort
      consensus_score  — score composite [0, 1] (plus élevé = plus sûr d'être bot)
    """

    def __init__(self, config: Optional[ConsensusScorerV2Config] = None,
                 enabled: bool = False):
        # Support double syntaxe : ConsensusScorerV2(enabled=True)
        if config is None:
            config = ConsensusScorerV2Config(enabled=enabled)
        self.cfg = config

    @classmethod
    def from_config(cls, cfg: dict) -> "ConsensusScorerV2":
        v2 = cfg.get("consensus_v2", {})
        c  = ConsensusScorerV2Config(
            enabled                = v2.get("enabled", False),
            weights                = v2.get("weights", dict(DEFAULT_WEIGHTS)),
            fallback_weight        = v2.get("fallback_weight", 1.0),
            bot_threshold          = v2.get("bot_threshold", 0.50),
            agreement_weight       = v2.get("agreement_weight", 0.25),
            confidence_weight      = v2.get("confidence_weight", 0.25),
            spread_penalty_weight  = v2.get("spread_penalty_weight", 0.20),
            spread_alarm_threshold = v2.get("spread_alarm_threshold", 0.40),
            structural_boost       = v2.get("structural_boost", 0.05),
            structural_threshold   = v2.get("structural_threshold", 0.65),
            v2_alpha               = v2.get("v2_alpha", 0.50),
        )
        return cls(c)

    def score(
        self,
        block_probas: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Calcule le score de consensus V2.

        Args:
            block_probas : {module_name: proba_array(n,)}

        Returns:
            pd.DataFrame avec colonnes :
                proba, proba_v1, consensus_score,
                n_agree, frac_agree, mean_confidence,
                max_spread, spread_score,
                has_structural_boost, structural_delta,
                v2_active
        """
        arrays = {k: v for k, v in block_probas.items()
                  if isinstance(v, np.ndarray) and len(v) > 0}
        if not arrays:
            raise ValueError("block_probas est vide")

        n = len(next(iter(arrays.values())))
        proba_base = self._weighted_avg(arrays, n)

        # V2 désactivée → retourner V1 avec colonnes diagnostics vides
        if not self.cfg.enabled:
            return pd.DataFrame({
                "proba":                proba_base,
                "proba_v1":             proba_base,
                "consensus_score":      np.nan,
                "n_agree":              np.nan,
                "frac_agree":           np.nan,
                "mean_confidence":      np.nan,
                "max_spread":           np.nan,
                "spread_score":         np.nan,
                "has_structural_boost": False,
                "structural_delta":     0.0,
                "v2_active":            False,
            })

        # ── Métriques de consensus ────────────────────────────
        thr       = self.cfg.bot_threshold
        arr_stack = np.stack(list(arrays.values()), axis=1)   # (n, m)
        m         = arr_stack.shape[1]

        # 1. N modules d'accord
        n_agree    = (arr_stack >= thr).astype(float).sum(axis=1)
        frac_agree = n_agree / max(m, 1)

        # 2. Confiance moyenne (distance normalisée à 0.5)
        mean_conf = (np.abs(arr_stack - 0.5) * 2).mean(axis=1)

        # 3. Écart max inter-modules
        max_spread = arr_stack.max(axis=1) - arr_stack.min(axis=1)
        sth        = self.cfg.spread_alarm_threshold
        # spread_score : 1 si pas de conflit, décroît si spread > sth
        spread_score = np.where(
            max_spread <= sth,
            1.0,
            np.clip(1.0 - (max_spread - sth) / max(1.0 - sth, 1e-9), 0.0, 1.0),
        )

        # 4. Bonus structurel
        boost_arr = np.zeros(n)
        for mod in STRUCTURAL_MODULES:
            if mod in arrays:
                boost_arr += (arrays[mod] >= self.cfg.structural_threshold).astype(float) * self.cfg.structural_boost
        boost_arr           = np.clip(boost_arr, 0.0, 0.10)
        has_structural_boost = boost_arr > 0

        # ── Consensus score composite ─────────────────────────
        aw = self.cfg.agreement_weight
        cw = self.cfg.confidence_weight
        sw = self.cfg.spread_penalty_weight
        remaining = max(1.0 - aw - cw - sw, 0.0)

        consensus_score = np.clip(
            aw * frac_agree
            + cw * mean_conf
            + sw * spread_score
            + remaining * proba_base
            + boost_arr,
            0.0, 1.0,
        )

        # ── Probabilité finale V2 ─────────────────────────────
        alpha   = self.cfg.v2_alpha
        proba_v2 = np.clip(
            alpha * proba_base + (1 - alpha) * consensus_score,
            0.0, 1.0,
        )

        logger.info(
            "ConsensusV2 — n=%d modules=%d mean_agree=%.2f mean_spread=%.4f",
            n, m, float(n_agree.mean()), float(max_spread.mean()),
        )

        return pd.DataFrame({
            "proba":                np.round(proba_v2, 6),
            "proba_v1":             np.round(proba_base, 6),
            "consensus_score":      np.round(consensus_score, 6),
            "n_agree":              n_agree.astype(int),
            "frac_agree":           np.round(frac_agree, 4),
            "mean_confidence":      np.round(mean_conf, 4),
            "max_spread":           np.round(max_spread, 4),
            "spread_score":         np.round(spread_score, 4),
            "has_structural_boost": has_structural_boost,
            "structural_delta":     np.round(boost_arr, 4),
            "v2_active":            True,
        })

    def _weighted_avg(self, arrays: dict, n: int) -> np.ndarray:
        w_sum, w_tot = np.zeros(n), 0.0
        for name, arr in arrays.items():
            w = self.cfg.weights.get(name, self.cfg.fallback_weight)
            w_sum += arr * w
            w_tot += w
        return np.clip(w_sum / max(w_tot, 1e-9), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# Raccourcis standalone
# ─────────────────────────────────────────────────────────────

def score_v1(block_probas: Dict[str, np.ndarray],
             weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """V1 : moyenne pondérée simple."""
    cfg = ConsensusScorerConfig(weights=weights or dict(DEFAULT_WEIGHTS))
    return ConsensusScorer(cfg).score(block_probas)


def score_v2(block_probas: Dict[str, np.ndarray],
             enabled: bool = False,
             weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    V2 : scoring enrichi.
    enabled=False par defaut — activer UNIQUEMENT apres validation benchmark.
    """
    cfg = ConsensusScorerV2Config(
        enabled = enabled,
        weights = weights or dict(DEFAULT_WEIGHTS),
    )
    return ConsensusScorerV2(cfg).score(block_probas)
