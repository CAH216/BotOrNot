# -*- coding: utf-8 -*-
"""
champion_config.py
==================
Configuration officielle gelée — Jour J.
NE PAS MODIFIER sans benchmark 5-gates complet et accord explicite.

═══════════════════════════════════════════════════════════════
  CHAMPIONS OFFICIELS (à utiliser le Jour J)
═══════════════════════════════════════════════════════════════
  EN Champion : mean=101.5 | p5=94.9 | FP_max=1
  FR Champion : mean=47.5  | p5=41.0 | FP_max=1
  Score compétition global (E1-E6+E30+E31) : 570

  EN Fallback  : mean≈99.5 | FP_max=1  (sans Veto — plus safe si crash)
  FR Fallback  : mean=45.5 | FP_max=1  (sans Synth — baseline propre)

Validé le : 2026-04-01
Simulation Jour J : scripts/competition_simulation.py
Benchmarks ref   : scripts/benchmark_residual_surgery.py (EN)
                   scripts/benchmark_fr_dual.py (FR)
═══════════════════════════════════════════════════════════════

Note technique : le champ users_average_z_score en e-17 doit être
  traité comme ≈0 (pas de cas spécial). fillna(0) suffit.
"""

# ─── Pipeline global (Feature Extraction) ─────────────────────────────────────────────────────────────
EN_MONOLITH_CONFIG = {
    "use_vas":               True,   # VAS features (validé)
    "use_lrh":               True,   # LRH features (validé)
    "use_lrh2":              True,   # LRH2 residual features (validé)
    "use_lrh3":              False,  # Rejeté par Tournament (-1.0 EN)
    "use_temporal_motifs":   False,
    "use_semantic_coherence": False,
    "use_content_repetition": True,  # Promu par le Tournament (+4.5 EN)
    "use_high_roi":          False,
    "use_register_invariance": False, # Retiré (trop instable sur le Seed final)
    "use_llm_hallucinations": True,  # Détecteur des "Persona Generative AI"
}

FR_MONOLITH_CONFIG = {
    "use_vas":               True,   # VAS features (validé)
    "use_lrh":               True,   # LRH features (validé)
    "use_lrh2":              True,
    "use_lrh3":              False,
    "use_temporal_motifs":   False,
    "use_semantic_coherence": False,
    "use_content_repetition": False,
    "use_high_roi":          True,   # Promu par le Tournament (+1.0 FR)
    "use_register_invariance": False, # Retiré (trop instable)
    "use_llm_hallucinations": True,
}

# Alias de compatibilité pour les anciens scripts R&D
MONOLITH_CONFIG = EN_MONOLITH_CONFIG.copy()

# ─── EN Champion (Jour J — utiliser en priorité) ─────────────────────────────────
# Validé : mean=101.5 | p5=94.9 | FP_max=1
# E1=104  E3=94  E5=92  E30=116
EN_MINER_CONFIG = {
    "use_veto":             True,   # Veto actif — protège humains accusés
    "use_expansion":        False,  # Expansion désactivée (instable E3)
    "proba_low":            0.01,
    "proba_high":           0.35,
    "forensic_percentile":  65,
    "human_archetype_cap":  0.30,
}
EN_COURT_CONFIG = {
    "k":              3,
    "min_bot_votes":  2,
}

# ─── EN Fallback (si champion crash ou anomalie) ─────────────────────────────────
# Mean≈99.5 | FP_max=1 | Sans Veto — comportement plus conservateur
EN_FALLBACK_MINER_CONFIG = {
    "use_veto":             False,  # Veto désactivé — moins de risk FP
    "use_expansion":        False,
    "proba_low":            0.01,
    "proba_high":           0.35,
    "forensic_percentile":  65,
    "human_archetype_cap":  0.30,
}
EN_FALLBACK_COURT_CONFIG = {
    "k":              3,
    "min_bot_votes":  2,
}

# ─── FR Champion (Jour J — utiliser en priorité) ─────────────────────────────────
# Validé : mean=47.5 | p5=41.0 | FP_max=1
# E2=54  E4=48  E6=34  E31=54
FR_MINER_CONFIG = {
    "use_veto":             False,  # Veto FR désactivé (trop sensible)
    "use_expansion":        False,
    "proba_low":            0.01,
    "proba_high":           0.50,
    "forensic_percentile":  50,
    "human_archetype_cap":  0.30,
}
FR_COURT_CONFIG = {
    "k":              3,
    "min_bot_votes":  3,  # Seuil strict FR
}

# ─── FR Fallback (si champion crash ou anomalie) ─────────────────────────────────
# Mean=45.5 | FP_max=1 | Sans Synth — baseline propre
FR_FALLBACK_MINER_CONFIG = {
    "use_veto":             False,
    "use_expansion":        False,
    "proba_low":            0.01,
    "proba_high":           0.50,
    "forensic_percentile":  50,
    "human_archetype_cap":  0.30,
}
FR_FALLBACK_COURT_CONFIG = {
    "k":              3,
    "min_bot_votes":  3,
}

# ─── Synthétique FR v2 ───────────────────────────────────────────────────────────
# Activé en training uniquement (jamais appliqué au dataset de test)
FR_SYNTH_CONFIG = {
    "enabled":       True,
    "n_per_bot":     50,
    "n_per_human":   60,
    "seed":          42,
    "bot_archetypes":   ["fr_midnighter_bot", "fr_poll_nationalist_bot", "fr_gentle_promo_bot"],
    "human_archetypes": ["fr_insomniac_human", "fr_political_human", "fr_lifestyle_fr_human"],
}

# ─── Events d'entraînement historiques ──────────────────────────────────────────
EN_TRAIN_EVENTS = [1, 3, 5, 30]   # Tous les EN connus → training Jour J
FR_TRAIN_EVENTS = [2, 4, 6, 31]   # Tous les FR connus → training Jour J

# ─── Helpers ─────────────────────────────────────────────────────────────────────

def get_en_pipeline(fallback: bool = False):
    from src.features.candidate_miner_court import CandidateMiner, PairwiseCourt
    if fallback:
        return CandidateMiner(**EN_FALLBACK_MINER_CONFIG), PairwiseCourt(**EN_FALLBACK_COURT_CONFIG)
    return CandidateMiner(**EN_MINER_CONFIG), PairwiseCourt(**EN_COURT_CONFIG)

def get_fr_pipeline(fallback: bool = False):
    from src.features.candidate_miner_court import CandidateMiner, PairwiseCourt
    if fallback:
        return CandidateMiner(**FR_FALLBACK_MINER_CONFIG), PairwiseCourt(**FR_FALLBACK_COURT_CONFIG)
    return CandidateMiner(**FR_MINER_CONFIG), PairwiseCourt(**FR_COURT_CONFIG)

def get_fr_synthetic_data():
    from scripts.benchmark_fr_dual import generate_fr_hard_negatives_v2
    cfg = FR_SYNTH_CONFIG
    return generate_fr_hard_negatives_v2(
        n_per_bot   = cfg["n_per_bot"],
        n_per_human = cfg["n_per_human"],
        seed        = cfg["seed"],
    )

def get_lgbm_params():
    return {"random_state": 42, "verbose": -1, "n_estimators": 150}

def get_kfold_params():
    return {"n_splits": 5, "shuffle": True, "random_state": 42}

# ─── Scores officiels de référence ───────────────────────────────────────────────
OFFICIAL_SCORES = {
    "EN": {
        "mean": 101.5, "p5": 94.9, "FP_max": 1,
        "E1": 104, "E3": 94, "E5": 92, "E30": 116,
        "config": "Champion EN — use_veto=True, proba_high=0.35, forensic_p=65",
    },
    "FR": {
        "mean": 47.5, "p5": 41.0, "FP_max": 1,
        "E2": 54, "E4": 48, "E6": 34, "E31": 54,
        "config": "Champion FR — Synth v2, proba_high=0.50, forensic_p=50",
    },
    "COMPETITION_TOTAL": {
        "score": 570,
        "EN_total": 384, "FR_total": 186,
        "recall":    94.9, "precision": 98.4,
        "note": "Simulation LOEO complète E1→E31 (2026-04-01)",
    },
    "FR_PREVIOUS": {
        "mean": 45.5, "p5": 37.0,
        "note": "Champion FR avant Synth v2 (fallback de référence)",
    },
}
