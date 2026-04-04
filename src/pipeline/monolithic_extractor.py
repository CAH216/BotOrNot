# -*- coding: utf-8 -*-
"""
monolithic_extractor.py
L'Usine d'Assemblage "Exploit Every Bit".
Combine toutes les features (Base, LLM, Incon, Hacker, V2) en une seule Dataframe ultra-large.
"""
import pandas as pd
import sys, os
# Assure que scripts/ est dans le path (competition_benchmark.py est dans scripts/)
if str(os.path.join(os.getcwd(), 'scripts')) not in sys.path:
    sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from competition_benchmark import extract_competition_features
from src.features.human_inconsistency import extract_human_inconsistency

# Validated Production Features
from src.features.ghost_human_protector_v2 import extract_ghost_human_protector_v2
_GHOST_SLIM_COLS = ["user_id", "gh_n_posts", "gh_len_cv", "gh_skeleton_unique_ratio"]
from src.features.content_repetition import extract_content_repetition

from src.features.time_delta_v2 import extract_time_delta_v2
from src.features.sentiment_volatility import extract_sentiment_volatility
from src.features.llm_hallucinations import extract_llm_hallucinations

# Validated Production Features
from src.features.ghost_human_protector_v2 import extract_ghost_human_protector_v2
_GHOST_SLIM_COLS = ["user_id", "gh_n_posts", "gh_len_cv", "gh_skeleton_unique_ratio"]


# Content Repetition — Jaccard entre posts consécutifs (signal template bot)
_CREP_COLS = ["user_id", "csr_jaccard_mean", "csr_jaccard_std",
              "csr_jaccard_min", "csr_unigram_pct_shared", "csr_template_score"]


def extract_monolithic_features(u_df: pd.DataFrame, p_df: pd.DataFrame, metadata: dict = None, config: dict = None) -> pd.DataFrame:
    """ Produit un vecteur de features hyper-dense pour LGBM. """
    if metadata is None:
        metadata = {}
    if config is None:
        config = {
            "use_high_roi": False,
            "use_content_repetition": False,
            "use_llm_hallucinations": False,
        }
        
    # 1. Base
    df_base = extract_competition_features(u_df, p_df)
    
    # 2. Expert Detectors Outputs
    df_incon = extract_human_inconsistency(u_df, p_df)
    # NOTE: llm_persona removed (not in prod pipeline — features always 0)
    
    # Merge chain
    merged = df_base.merge(df_incon, on="user_id", how="left").fillna(0)

    # Ghost Slim Validated (KEEP — EN +1pt, 0 FP) — EN 86.0 / FR 36.0
    df_ghost_slim = extract_ghost_human_protector_v2(u_df, p_df)
    available = [c for c in _GHOST_SLIM_COLS if c in df_ghost_slim.columns]
    merged = merged.merge(df_ghost_slim[available], on="user_id", how="left").fillna(0)

    # ── LLM Hallucinations [OFF par défaut]
    if config.get("use_llm_hallucinations", False):
        df_llm = extract_llm_hallucinations(u_df, p_df)
        merged = merged.merge(df_llm, on="user_id", how="left").fillna(0)

    if config.get("use_high_roi", False):
        df_delta = extract_time_delta_v2(u_df, p_df)
        df_sent = extract_sentiment_volatility(u_df, p_df)
        merged = merged.merge(df_delta, on="user_id", how="left").fillna(0)
        merged = merged.merge(df_sent, on="user_id", how="left").fillna(0)
        
    # ── Content Repetition [OFF par défaut] — Jaccard consec. posts (template bot signal)
    if config.get("use_content_repetition", False):
        df_crep = extract_content_repetition(u_df, p_df)
        avail   = [c for c in _CREP_COLS if c in df_crep.columns]
        merged  = merged.merge(df_crep[avail], on="user_id", how="left").fillna(0)

    return merged
