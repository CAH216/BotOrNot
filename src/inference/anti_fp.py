# -*- coding: utf-8 -*-
"""
src/inference/anti_fp.py
--------------------------
Logique anti-faux-positifs — Protection des humains atypiques
==============================================================
Ce module implémente des règles de prudence qui ajustent les
probabilités brutes produites par les modèles avant d'appliquer
le seuil final.

Principes :
  1. Un seul module qui signale « bot » ne suffit pas (pénalité).
  2. Les power-users (forts followers, vérifiés) bénéficient d'une
     protection accrue.
  3. Des règles de conflit explicites détectent les incohérences
     entre signaux et réduisent la confiance.

Usage :
    from src.inference.anti_fp import AntiFPFilter

    af = AntiFPFilter.from_config(cfg)
    proba_adjusted = af.apply(proba_df, feature_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuration par défaut (compatible avec golden_baseline.yaml)
# ─────────────────────────────────────────────────────────────

@dataclass
class AntiFPConfig:
    """Paramètres du filtre anti-faux-positifs."""
    enabled: bool                    = True
    min_modules_for_bot: int         = 2        # min modules signalant bot
    unilateral_penalty: float        = 0.10     # réduction si seul 1 module
    # Power-user protection
    power_user_protection: bool      = True
    pu_min_followers: int            = 5_000
    pu_follower_penalty: float       = 0.05
    pu_verified_penalty: float       = 0.08     # si compte vérifié
    # Conflit de signaux
    conflict_rules_enabled: bool     = True
    conflict_penalty: float          = 0.08     # pénalité si conflit détecté
    # Seuil de confiance minimal pour considérer qu'un module "signale bot"
    bot_signal_threshold: float      = 0.50


# ─────────────────────────────────────────────────────────────
# Moteur de règles
# ─────────────────────────────────────────────────────────────

class AntiFPFilter:
    """
    Filtre post-modèle qui ajuste les probabilités en appliquant
    des règles de prudence pour protéger les humains atypiques.

    Paramètres :
        config : AntiFPConfig

    Exemple :
        af = AntiFPFilter(AntiFPConfig(min_modules_for_bot=2))
        adjusted = af.apply(proba_df, feature_df)
    """

    def __init__(self, config: Optional[AntiFPConfig] = None):
        self.cfg = config or AntiFPConfig()

    @classmethod
    def from_config(cls, cfg: dict) -> "AntiFPFilter":
        """Construit depuis un dict YAML (section anti_fp)."""
        if not cfg.get("enabled", True):
            return cls(AntiFPConfig(enabled=False))
        pu = cfg.get("power_user_protection", {})
        return cls(AntiFPConfig(
            enabled               = cfg.get("enabled", True),
            min_modules_for_bot   = cfg.get("min_modules_for_bot", 2),
            unilateral_penalty    = cfg.get("unilateral_penalty", 0.10),
            power_user_protection = pu.get("enabled", True) if isinstance(pu, dict) else True,
            pu_min_followers      = pu.get("min_followers", 5_000) if isinstance(pu, dict) else 5_000,
            pu_follower_penalty   = pu.get("follower_penalty", 0.05) if isinstance(pu, dict) else 0.05,
        ))

    # ── Point d'entrée principal ──────────────────────────────

    def apply(
        self,
        proba_df: pd.DataFrame,
        feature_df: Optional[pd.DataFrame] = None,
        block_probas: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Applique les règles anti-FP sur les probabilités.

        Args:
            proba_df    : DataFrame avec colonnes ['account_id', 'proba']
                          (ou une seule colonne float in [0,1])
            feature_df  : DataFrame de features (pour les règles par-compte)
            block_probas: dict {'tabular': proba_array, 'temporal': proba_array, …}
                          Si fourni, permet de détecter les signaux unilatéraux.

        Returns:
            DataFrame identique à proba_df avec colonne 'proba_adjusted'
            et colonnes de diagnostic.
        """
        if not self.cfg.enabled:
            proba_df = proba_df.copy()
            proba_df["proba_adjusted"] = proba_df["proba"]
            proba_df["anti_fp_triggered"] = False
            return proba_df

        out = proba_df.copy()
        p   = out["proba"].values.copy().astype(float)
        triggered = np.zeros(len(p), dtype=bool)
        reasons   = [""] * len(p)

        def _add_reason(mask, reason):
            for i in np.where(mask)[0]:
                reasons[i] = (reasons[i] + ";" + reason).lstrip(";")

        # ── Règle 1 : Signal unilatéral ──────────────────────
        if block_probas and self.cfg.min_modules_for_bot > 1:
            thr = self.cfg.bot_signal_threshold
            # Compter combien de modules signalent bot pour chaque compte
            n_signaling = sum(
                (arr >= thr).astype(int)
                for arr in block_probas.values()
                if isinstance(arr, np.ndarray) and len(arr) == len(p)
            )
            unilateral_mask = (n_signaling == 1) & (p >= thr)
            if unilateral_mask.any():
                penalty = self.cfg.unilateral_penalty
                p[unilateral_mask] = np.clip(p[unilateral_mask] - penalty, 0.0, 1.0)
                triggered |= unilateral_mask
                _add_reason(unilateral_mask, f"unilateral_signal(-{penalty:.2f})")
                logger.info(
                    "AntiFP — %d comptes pénalisés pour signal unilatéral",
                    unilateral_mask.sum()
                )

        # ── Règle 2 : Power-user protection ──────────────────
        if self.cfg.power_user_protection and feature_df is not None:
            followers_col = next(
                (c for c in feature_df.columns if "followers" in c.lower()
                 and "log" not in c.lower() and "ratio" not in c.lower()
                 and "extreme" not in c.lower()),
                None,
            )
            if followers_col:
                followers = feature_df[followers_col].fillna(0).values
                pu_mask = followers >= self.cfg.pu_min_followers
                if pu_mask.any():
                    penalty = self.cfg.pu_follower_penalty
                    p[pu_mask] = np.clip(p[pu_mask] - penalty, 0.0, 1.0)
                    triggered |= pu_mask
                    _add_reason(pu_mask, f"power_user_followers(-{penalty:.2f})")
                    logger.info(
                        "AntiFP — %d comptes power-user protégés (followers≥%d)",
                        pu_mask.sum(), self.cfg.pu_min_followers
                    )

            # Vérifié → pénalité supplémentaire
            verified_col = next(
                (c for c in feature_df.columns if "verified" in c.lower()), None
            )
            if verified_col:
                verified_mask = feature_df[verified_col].fillna(0).values.astype(bool)
                if verified_mask.any():
                    penalty = self.cfg.pu_verified_penalty
                    p[verified_mask] = np.clip(p[verified_mask] - penalty, 0.0, 1.0)
                    triggered |= verified_mask
                    _add_reason(verified_mask, f"verified_account(-{penalty:.2f})")
                    logger.info(
                        "AntiFP — %d comptes vérifiés protégés", verified_mask.sum()
                    )

        # ── Règle 3 : Conflit de signaux détecté ─────────────
        if self.cfg.conflict_rules_enabled and block_probas:
            conflict_mask = self._detect_conflicts(block_probas, p)
            if conflict_mask.any():
                penalty = self.cfg.conflict_penalty
                p[conflict_mask] = np.clip(p[conflict_mask] - penalty, 0.0, 1.0)
                triggered |= conflict_mask
                _add_reason(conflict_mask, f"signal_conflict(-{penalty:.2f})")
                logger.info(
                    "AntiFP — %d comptes avec conflit de signaux", conflict_mask.sum()
                )

        # ── Assemblage final ──────────────────────────────────
        out["proba_adjusted"]    = np.round(p, 6)
        out["anti_fp_triggered"] = triggered
        out["anti_fp_reason"]    = reasons
        out["proba_delta"]       = np.round(
            out["proba_adjusted"] - out["proba"], 6
        )

        n_affected = triggered.sum()
        logger.info(
            "AntiFP — %d/%d comptes ajustés (delta moyen: %.4f)",
            n_affected, len(p),
            float(out.loc[triggered, "proba_delta"].mean()) if n_affected else 0.0,
        )
        return out

    # ── Détecteur de conflits ─────────────────────────────────

    def _detect_conflicts(
        self, block_probas: dict, p: np.ndarray
    ) -> np.ndarray:
        """
        Détecte les conflits logiques entre les signaux des modules.

        Conflit = un module signal très botlike (> 0.75) ET
                  un autre module signal très humanlike (< 0.25)
                  sur le même compte, quand la proba globale est élevée.
        """
        if len(block_probas) < 2:
            return np.zeros(len(p), dtype=bool)

        arrays = {
            k: v for k, v in block_probas.items()
            if isinstance(v, np.ndarray) and len(v) == len(p)
        }
        if len(arrays) < 2:
            return np.zeros(len(p), dtype=bool)

        conflict = np.zeros(len(p), dtype=bool)
        keys = list(arrays.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = arrays[keys[i]], arrays[keys[j]]
                # Fort signal bot dans A, fort signal humain dans B (ou vice-versa)
                c1 = (a > 0.75) & (b < 0.25)
                c2 = (a < 0.25) & (b > 0.75)
                # Ne marque conflit que si proba globale déjà haute (risque de FP)
                conflict |= (c1 | c2) & (p > self.cfg.bot_signal_threshold)
        return conflict


# ─────────────────────────────────────────────────────────────
# Utilitaires standalone
# ─────────────────────────────────────────────────────────────

def apply_conservative_rules(
    proba_df: pd.DataFrame,
    feature_df: Optional[pd.DataFrame] = None,
    block_probas: Optional[dict] = None,
) -> pd.DataFrame:
    """Raccourci : applique les règles anti-FP en mode conservateur."""
    cfg = AntiFPConfig(
        enabled=True,
        min_modules_for_bot=2,
        unilateral_penalty=0.15,
        power_user_protection=True,
        pu_min_followers=3_000,
        pu_follower_penalty=0.08,
        pu_verified_penalty=0.10,
        conflict_rules_enabled=True,
        conflict_penalty=0.10,
    )
    return AntiFPFilter(cfg).apply(proba_df, feature_df, block_probas)


def apply_balanced_rules(
    proba_df: pd.DataFrame,
    feature_df: Optional[pd.DataFrame] = None,
    block_probas: Optional[dict] = None,
) -> pd.DataFrame:
    """Raccourci : applique les règles anti-FP en mode balanced."""
    cfg = AntiFPConfig(
        enabled=True,
        min_modules_for_bot=1,
        unilateral_penalty=0.08,
        power_user_protection=True,
        pu_min_followers=10_000,
        pu_follower_penalty=0.04,
        pu_verified_penalty=0.05,
        conflict_rules_enabled=True,
        conflict_penalty=0.06,
    )
    return AntiFPFilter(cfg).apply(proba_df, feature_df, block_probas)


def apply_aggressive_rules(
    proba_df: pd.DataFrame,
    feature_df: Optional[pd.DataFrame] = None,
    block_probas: Optional[dict] = None,
) -> pd.DataFrame:
    """Raccourci : aucune règle anti-FP (mode recall maximal)."""
    cfg = AntiFPConfig(enabled=False)
    return AntiFPFilter(cfg).apply(proba_df, feature_df, block_probas)
