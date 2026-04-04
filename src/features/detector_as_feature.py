# -*- coding: utf-8 -*-
"""
src/features/detector_as_feature.py

Detector-as-Feature Integration V2
===================================
Prend les sorties brutes de chaque sous-détecteur et produit :
  1. Un score composite par détecteur (0-1)
  2. Un flag de confiance (flag_* = 1 si le score dépasse un seuil "certain")
  3. Une marge de confiance (margin_* = distance au seuil de décision)
  4. Des signaux d'interaction croisée entre détecteurs

RÈGLE ABSOLUE : Aucun vote / committee ici.
Ce module produit uniquement un DataFrame de features numériques
que LightGBM/CatBoost apprendront à combiner librement.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.human_inconsistency import extract_human_inconsistency
from src.features_lab.llm_persona import extract_llm_persona
from src.features_lab.template_madlibs import extract_template_madlibs
from src.features_lab.clock_forensics import extract_clock_forensics
from src.features_lab.ghost_protector_v2 import extract_ghost_protector_v2


# ─── Score builders ────────────────────────────────────────────────────────────

def _incon_score(df_incon: pd.DataFrame) -> pd.Series:
    """
    human_inconsistency : incohérence entre username/name/bio.
    Score bot = 1 - (signe de cohérence humaine).
    Les deux colonnes _x/_y couvrent respectivement username/bio et name/bio.
    """
    lr  = df_incon.get("human_length_ratio_x", pd.Series(dtype=float)).fillna(0.5)
    lex = df_incon.get("human_lexical_entropy_x", pd.Series(dtype=float)).fillna(0.0)
    # Un bot a une longueur ratio extrême (proche de 0 ou >5) et faible entropie
    ratio_extreme = (lr - 1.0).abs()  # 0 = normal, élevé = suspect
    score = (ratio_extreme / (ratio_extreme.max() + 1e-9)) * 0.5 + (1 - lex.clip(0, 1)) * 0.5
    return score.clip(0, 1)


def _llm_score(df_llm: pd.DataFrame) -> pd.Series:
    """
    LLM persona : 4 signaux combinés linéairement.
    llm_bio_vanilla + llm_style_smoothness + llm_zero_noise + llm_buzzword_density
    Chacun déjà normalisé 0-1 par son extracteur.
    """
    cols = ["llm_bio_vanilla", "llm_style_smoothness", "llm_zero_noise", "llm_buzzword_density"]
    weights = [0.35, 0.25, 0.20, 0.20]
    score = sum(
        df_llm.get(c, pd.Series(dtype=float)).fillna(0.0) * w
        for c, w in zip(cols, weights)
    )
    return score.clip(0, 1)


def _madlibs_score(df_mad: pd.DataFrame) -> pd.Series:
    """
    Madlibs : madlib_clone_count normalisé + flag madlib_is_cloned.
    """
    cnt = df_mad.get("madlib_clone_count", pd.Series(dtype=float)).fillna(0.0)
    cloned = df_mad.get("madlib_is_cloned", pd.Series(dtype=float)).fillna(0.0)
    # Clone count : 0 = unique, 1-2 = suspect, 3+ = bot certain
    score = (cnt / (cnt.max() + 1e-9)) * 0.6 + cloned * 0.4
    return score.clip(0, 1)


def _clock_score(df_clk: pd.DataFrame) -> pd.Series:
    """
    Clock forensics : utilise les colonnes globales (sans suffix _x / _y).
    clock_frac_00 = fraction des posts avec seconde = 00 (API timer)
    clock_frac_mod5 = fraction modulo 5 (bot scheduler)
    clock_std_sec = faible std → pattern mécanique
    clock_is_api = flag dur
    """
    f00   = df_clk.get("clock_frac_00",   pd.Series(dtype=float)).fillna(0.0)
    fmod5 = df_clk.get("clock_frac_mod5", pd.Series(dtype=float)).fillna(0.0)
    std   = df_clk.get("clock_std_sec",   pd.Series(dtype=float)).fillna(30.0)
    api   = df_clk.get("clock_is_api",    pd.Series(dtype=float)).fillna(0.0)
    # Normalise std : faible std = suspect (inverse)
    std_norm = 1.0 - (std / (std.max() + 1e-9)).clip(0, 1)
    score = f00 * 0.35 + fmod5 * 0.25 + std_norm * 0.20 + api * 0.20
    return score.clip(0, 1)


def _protector_score(df_prot: pd.DataFrame) -> pd.Series:
    """
    Ghost protector V2 : protect_variance / protect_noise / protect_typo.
    Un vrai humain a variance élevée, noise élevé et typos.
    On inverse → protector_bot_score = 0 si l'humain est clairement humain.
    """
    var   = df_prot.get("protect_variance", pd.Series(dtype=float)).fillna(0.0)
    noise = df_prot.get("protect_noise",    pd.Series(dtype=float)).fillna(0.0)
    typo  = df_prot.get("protect_typo",     pd.Series(dtype=float)).fillna(0.0)
    # Le "score du protector" = signal d'innocence humaine (élevé = probablement humain)
    # En tant que feature, on le laisse brut (LightGBM apprend le sens)
    human_innocence = (var + noise + typo) / 3.0
    return human_innocence.clip(0, 1)


# ─── Extracteur principal ──────────────────────────────────────────────────────

def extract_detector_as_feature(
    u_df: pd.DataFrame,
    p_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retourne un DataFrame indexé sur user_id avec :
      - score brut de chaque détecteur
      - flag (1 si score > seuil « fort »)
      - margin (distance au seuil)
      - signaux d'interaction croisée
    """
    if u_df.empty:
        return pd.DataFrame()

    uid = u_df["user_id"].values

    # ── 1. Appel des détecteurs bruts ─────────────────────────────────────────
    try:
        df_incon = extract_human_inconsistency(u_df, p_df)
    except Exception:
        df_incon = pd.DataFrame({"user_id": uid})

    try:
        df_llm = extract_llm_persona(u_df, p_df)
    except Exception:
        df_llm = pd.DataFrame({"user_id": uid})

    try:
        df_mad = extract_template_madlibs(u_df, p_df)
    except Exception:
        df_mad = pd.DataFrame({"user_id": uid})

    try:
        df_clk = extract_clock_forensics(u_df, p_df)
    except Exception:
        df_clk = pd.DataFrame({"user_id": uid})

    try:
        df_prot = extract_ghost_protector_v2(u_df, p_df)
    except Exception:
        df_prot = pd.DataFrame({"user_id": uid})

    # ── 2. Scores scalaires composites ────────────────────────────────────────
    incon_s   = _incon_score(df_incon).values
    llm_s     = _llm_score(df_llm).values
    madlibs_s = _madlibs_score(df_mad).values
    clock_s   = _clock_score(df_clk).values
    protector_s = _protector_score(df_prot).values  # signal d'innocence

    # ── 3. Flags & Marges ────────────────────────────────────────────────────
    # Seuils calibrés de manière conservatrice (jamais comme décideurs)
    THRESH_INCON   = 0.60
    THRESH_LLM     = 0.55
    THRESH_MADLIBS = 0.50
    THRESH_CLOCK   = 0.50

    res = pd.DataFrame({"user_id": uid})

    # Scores bruts
    res["det_incon_score"]    = incon_s
    res["det_llm_score"]      = llm_s
    res["det_madlibs_score"]  = madlibs_s
    res["det_clock_score"]    = clock_s
    res["det_protector_score"] = protector_s  # innocence humaine (inversé)

    # Flags durs
    res["det_flag_incon"]    = (incon_s   >= THRESH_INCON).astype(np.float32)
    res["det_flag_llm"]      = (llm_s     >= THRESH_LLM).astype(np.float32)
    res["det_flag_madlibs"]  = (madlibs_s >= THRESH_MADLIBS).astype(np.float32)
    res["det_flag_clock"]    = (clock_s   >= THRESH_CLOCK).astype(np.float32)

    # Marges (distance au seuil → utile pour la confiance)
    res["det_margin_incon"]   = incon_s   - THRESH_INCON
    res["det_margin_llm"]     = llm_s     - THRESH_LLM
    res["det_margin_madlibs"] = madlibs_s - THRESH_MADLIBS
    res["det_margin_clock"]   = clock_s   - THRESH_CLOCK

    # ── 4. Signaux d'interaction croisée (pour LightGBM) ─────────────────────
    # Combien de détecteurs sonnent en même temps ?
    flag_sum = (
        res["det_flag_incon"] +
        res["det_flag_llm"] +
        res["det_flag_madlibs"] +
        res["det_flag_clock"]
    )
    res["det_flag_count"]   = flag_sum           # 0-4 détecteurs actifs
    res["det_consensus"]    = flag_sum >= 3       # consensus fort (≥3)
    res["det_consensus"]    = res["det_consensus"].astype(np.float32)
    res["det_consensus_2"]  = (flag_sum >= 2).astype(np.float32)  # consensus modéré

    # Score global moyen (soft ensemble — feature, pas décideur)
    res["det_mean_score"] = (incon_s + llm_s + madlibs_s + clock_s) / 4.0

    # Interaction clock × llm (bots API avec persona lisse = doublement suspect)
    res["det_clock_x_llm"] = clock_s * llm_s

    # Interaction madlibs × incon (clone de bio avec username incohérent)
    res["det_madlibs_x_incon"] = madlibs_s * incon_s

    # Protecteur vs consensus (humain fantôme avec signal clock fort = faux positif potentiel)
    res["det_protector_vs_clock"] = protector_s * clock_s

    # ── 5. Nettoyage final ────────────────────────────────────────────────────
    for c in res.columns:
        if c != "user_id":
            res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0.0)

    return res
