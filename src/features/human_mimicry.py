# -*- coding: utf-8 -*-
"""
human_mimicry.py — Mimicry Slim (Golden Core v2)
=================================================
Audit SHAP (31-03-2026) : 90% des variables originales avaient 0 splits dans LightGBM.
Seules 4 variables portent un signal orthogonal réel.

Golden Core conservé :
  1. hm_vanilla_with_long_posts   — Bio générique × Verbosité extrême
  2. hm_low_night_high_human      — Diurne parfait × Bio trop propre
  3. hm_interaction_night_focus   — Nuit × Topic Focus (incohérence timing)
  4. hm_vanilla_with_topic_focus  — Bio vide × Sujet unique d'Event

Tout le reste (typos, hashtags, gating, confirmation, cliche) est supprimé.
"""

import pandas as pd
import numpy as np


def extract_human_mimicry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Injecte les 4 features du Golden Core directement dans le DataFrame merged.
    Entrée  : DataFrame merged avec au moins les colonnes de base (txt_avg_len, etc.)
    Sortie  : même DataFrame + 4 colonnes hm_*
    """
    res = df.copy()

    # ── Résolution safe des colonnes sources ──────────────────────────────────
    def _col(name, default):
        return res[name] if name in res.columns else pd.Series(default, index=res.index, dtype=float)

    z_vanilla  = _col("llm_bio_vanilla",          0.0)
    z_len      = _col("txt_avg_len",              100.0)
    z_night    = _col("tmp_night_ratio",            0.25)
    z_relfocus = _col("rel_topic_focus",            0.0)

    # ── 1. Bio générique × Verbosité ─────────────────────────────────────────
    # Un humain qui écrit beaucoup a généralement un profil élaboré.
    # Un bot LLM produit de longs posts avec une bio vide ou générique.
    res["hm_vanilla_with_long_posts"] = z_vanilla * z_len

    # ── 2. Diurne parfait × Bio générique ────────────────────────────────────
    # Un humain digne de ce nom dort la nuit. Un bot 9h-17h n'a jamais de posts nocturnes
    # ET affiche souvent une bio trop lisse (générée).
    diurnal_focus = (1.0 - z_night).clip(0, 1)
    res["hm_low_night_high_human"] = diurnal_focus * z_vanilla

    # ── 3. Nuit × Topic Focus ─────────────────────────────────────────────────
    # Un humain actif la nuit parle de tout. Un bot programmé pour l'Event
    # poste 100% sur le sujet même à 3h du matin : le croisement révèle la mécanique.
    res["hm_interaction_night_focus"] = z_night * z_relfocus

    # ── 4. Bio vide × Sujet unique ────────────────────────────────────────────
    # Profil sans âme + verrouillage thématique total = le combo bot narratif classique.
    res["hm_vanilla_with_topic_focus"] = z_vanilla * z_relfocus

    return res
