# -*- coding: utf-8 -*-
"""
src/data/adapters/coverage.py
-----------------------------
Générateur de rapport standardisé pour vérifier la qualité 
et la couverture d'un dataset ingéré (via un DataBundle).
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from src.data.schema import DataBundle, AccountCols, PostCols, EdgeCols, LabelCols


def generate_coverage_report(bundle: DataBundle) -> Dict[str, Any]:
    """
    Analyse le DataBundle et retourne des métriques de couverture.
    """
    report = {
        "dataset_source": bundle.source_path,
        "format": bundle.source_format,
        "total_accounts": bundle.n_accounts,
        "total_posts": bundle.n_posts,
        "coverage": {}
    }
    
    if bundle.n_accounts == 0:
        return report

    # Couverture Account Metrics
    act_df = bundle.accounts_df
    n_act = len(act_df)
    
    def _cov(col, df):
        if df is None or col not in df.columns:
            return 0.0
        return (df[col].notna() & (df[col] != "")).mean()
        
    report["coverage"]["accounts"] = {
        "bio": _cov(AccountCols.BIO, act_df),
        "location": _cov(AccountCols.LOCATION, act_df),
        "created_at": _cov(AccountCols.CREATED_AT, act_df),
        "followers_metrics": _cov(AccountCols.FOLLOWERS, act_df)
    }

    # Couverture Posts
    if bundle.n_posts > 0 and bundle.posts_df is not None:
        pst = bundle.posts_df
        report["coverage"]["posts"] = {
            "text": _cov(PostCols.TEXT, pst),
            "created_at": _cov(PostCols.CREATED_AT, pst),
            "source": _cov(PostCols.SOURCE, pst),
            "interactions (mentions/hashtags)": _cov(PostCols.HASHTAGS, pst)
        }
    else:
        report["coverage"]["posts"] = "No Posts Extracted"

    # Couverture Relationnelle
    if bundle.edges_df is not None and not bundle.edges_df.empty:
        report["coverage"]["relational"] = {
            "total_edges": len(bundle.edges_df),
            "unique_sources": bundle.edges_df[EdgeCols.SOURCE].nunique() if EdgeCols.SOURCE in bundle.edges_df.columns else 0,
            "unique_targets": bundle.edges_df[EdgeCols.TARGET].nunique() if EdgeCols.TARGET in bundle.edges_df.columns else 0
        }
    else:
        report["coverage"]["relational"] = "No Relational Graph Extracted"

    # Couverture Labels
    if bundle.labels_df is not None and LabelCols.LABEL in bundle.labels_df.columns:
        lbl = bundle.labels_df
        count_bots = (lbl[LabelCols.LABEL] == 1).sum()
        count_humans = (lbl[LabelCols.LABEL] == 0).sum()
        total_lbl = count_bots + count_humans
        report["coverage"]["labels"] = {
            "labeled_ratio": total_lbl / n_act if n_act > 0 else 0,
            "bot_ratio": count_bots / total_lbl if total_lbl > 0 else 0
        }
    else:
        report["coverage"]["labels"] = "Unlabeled Dataset"

    return report
