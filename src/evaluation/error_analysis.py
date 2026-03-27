# -*- coding: utf-8 -*-
"""
src/evaluation/error_analysis.py
----------------------------------
Analyse d'erreurs du pipeline BotOrNot.

Objectif :
    Comprendre POURQUOI le modèle se trompe pour prioriser les améliora-
    tions de façon rationnelle plutôt que d'optimiser à l'aveugle.

Catégories d'erreurs :
    - Faux positifs (FP) : humains classés comme bots → impact sur la confiance
    - Faux négatifs (FN) : bots non détectés → réel danger compétitif
    - Cas ambigus       : prob dans la zone grise, faciles à corriger avec
                          plus de données ou un meilleur seuil

Fonctions publiques :
    ErrorAnalyzer.analyze()       → DataFrame annoté complet
    ErrorAnalyzer.false_positives() / false_negatives() → top N cas
    ErrorAnalyzer.ambiguous_cases()  → cas dans la zone grise
    ErrorAnalyzer.score_distribution() → stats par catégorie
    ErrorAnalyzer.feature_importance_on_errors() → features qui causent les erreurs
    ErrorAnalyzer.report()        → rapport textuel synthétique

Usage :
    from src.evaluation.error_analysis import ErrorAnalyzer

    ea = ErrorAnalyzer(threshold=0.5, uncertainty_margin=0.12)
    analyzed = ea.analyze(y_true, y_prob, X=X_val, account_ids=ids)
    print(ea.report(analyzed))
    fp = ea.false_positives(analyzed, top_n=10)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Catégories d'erreur
_TP  = "true_positive"
_TN  = "true_negative"
_FP  = "false_positive"
_FN  = "false_negative"
_AMB = "ambiguous"


# ---------------------------------------------------------------------------
# ErrorAnalyzer
# ---------------------------------------------------------------------------

class ErrorAnalyzer:
    """
    Analyseur d'erreurs de classification pour BotOrNot.

    Args:
        threshold          : seuil de décision binaire
        uncertainty_margin : demi-largeur de la zone grise
    """

    def __init__(
        self,
        threshold:          float = 0.5,
        uncertainty_margin: float = 0.12,
    ) -> None:
        self.threshold          = threshold
        self.uncertainty_margin = uncertainty_margin
        self.low_bound  = max(0.0, threshold - uncertainty_margin)
        self.high_bound = min(1.0, threshold + uncertainty_margin)

    # ------------------------------------------------------------------
    # Analyse principale
    # ------------------------------------------------------------------

    def analyze(
        self,
        y_true:      np.ndarray,
        y_prob:      np.ndarray,
        X:           Optional[pd.DataFrame] = None,
        account_ids: Optional[pd.Series]    = None,
    ) -> pd.DataFrame:
        """
        Annote chaque exemple avec sa catégorie d'erreur et sa confiance.

        Args:
            y_true      : labels vrais (0/1)
            y_prob      : probabilités prédites
            X           : matrice de features (pour l'analyse par feature)
            account_ids : identifiants de compte

        Returns:
            DataFrame avec colonnes :
                account_id, y_true, y_prob, y_pred,
                error_type, confidence, margin_from_threshold,
                is_uncertain, rank_confidence
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= self.threshold).astype(int)

        # Catégorie d'erreur
        error_type = np.where(
            (y_pred == 1) & (y_true == 1), _TP,
            np.where((y_pred == 0) & (y_true == 0), _TN,
            np.where((y_pred == 1) & (y_true == 0), _FP, _FN))
        )

        # Cas ambigus (remplace FP/FN dans la zone grise)
        in_grey = (y_prob >= self.low_bound) & (y_prob <= self.high_bound)
        error_type = np.where(in_grey & (error_type.isin([_FP, _FN])
                              if False else np.isin(error_type, [_FP, _FN])),
                              _AMB, error_type)

        # Confiance : distance au seuil (plus grand = plus confiant)
        margin = np.abs(y_prob - self.threshold)
        confidence_score = margin / max(max(self.threshold,
                                            1 - self.threshold), 1e-9)

        # Niveau de confiance catégoriel
        conf_level = np.where(
            margin > self.uncertainty_margin * 2, "high",
            np.where(margin > self.uncertainty_margin,   "medium", "low")
        )

        n = len(y_true)
        ids = account_ids.reset_index(drop=True).values if account_ids is not None \
              else np.arange(n)

        df = pd.DataFrame({
            "account_id":            ids,
            "y_true":                y_true,
            "y_prob":                np.round(y_prob, 4),
            "y_pred":                y_pred,
            "error_type":            error_type,
            "confidence_level":      conf_level,
            "confidence_score":      np.round(confidence_score, 4),
            "margin_from_threshold": np.round(margin, 4),
            "is_uncertain":          in_grey,
        })

        # Rang de confiance dans chaque catégorie (1 = plus confiant)
        df["rank_in_category"] = (
            df.groupby("error_type")["confidence_score"]
              .rank(ascending=False, method="first")
              .astype(int)
        )

        # Attacher les features si fournies
        if X is not None:
            X_reset = X.reset_index(drop=True)
            df = pd.concat([df, X_reset], axis=1)

        logger.info(
            "ErrorAnalyzer : %d TP | %d TN | %d FP | %d FN | %d ambigus",
            (error_type == _TP).sum(), (error_type == _TN).sum(),
            (error_type == _FP).sum(), (error_type == _FN).sum(),
            (error_type == _AMB).sum(),
        )
        return df

    # ------------------------------------------------------------------
    # Filtres rapides
    # ------------------------------------------------------------------

    def false_positives(
        self,
        analyzed:  pd.DataFrame,
        top_n:     int = 20,
        sort_by:   str = "y_prob",   # prob la plus haute = le plus "sûr" d'être bot à tort
    ) -> pd.DataFrame:
        """
        Retourne les top N faux positifs (humains classés bots).

        Triés par probabilité décroissante : les plus "confiants" en erreur
        en premier → cas les plus urgents à corriger.
        """
        mask = analyzed["error_type"].isin([_FP, _AMB]) & (analyzed["y_true"] == 0)
        fp   = analyzed[mask].sort_values(sort_by, ascending=False)
        return fp.head(top_n).reset_index(drop=True)

    def false_negatives(
        self,
        analyzed: pd.DataFrame,
        top_n:    int = 20,
        sort_by:  str = "y_prob",   # prob la plus basse = bot le plus "caché"
    ) -> pd.DataFrame:
        """
        Retourne les top N faux négatifs (bots non détectés).

        Triés par probabilité croissante : les bots avec les scores les plus
        bas → les plus difficiles à attraper, priorité d'analyse.
        """
        mask = analyzed["error_type"].isin([_FN, _AMB]) & (analyzed["y_true"] == 1)
        fn   = analyzed[mask].sort_values(sort_by, ascending=True)
        return fn.head(top_n).reset_index(drop=True)

    def ambiguous_cases(
        self,
        analyzed: pd.DataFrame,
        top_n:    int = 30,
    ) -> pd.DataFrame:
        """
        Retourne les cas dans la zone grise (prob entre low et high bound).
        Triés par distance au seuil croissante (plus proche = plus ambigu).
        """
        mask = analyzed["error_type"] == _AMB
        return (analyzed[mask]
                .sort_values("margin_from_threshold", ascending=True)
                .head(top_n)
                .reset_index(drop=True))

    # ------------------------------------------------------------------
    # Distribution des scores
    # ------------------------------------------------------------------

    def score_distribution(self, analyzed: pd.DataFrame) -> pd.DataFrame:
        """
        Statistiques de distribution des scores par catégorie d'erreur.

        Returns:
            DataFrame avec count, mean, std, min, p25, median, p75, max
            pour chaque error_type.
        """
        stats = (
            analyzed.groupby("error_type")["y_prob"]
            .agg(
                count  = "count",
                mean   = "mean",
                std    = "std",
                min    = "min",
                p25    = lambda x: x.quantile(0.25),
                median = "median",
                p75    = lambda x: x.quantile(0.75),
                max    = "max",
            )
            .round(4)
            .reset_index()
        )
        # Ordre logique
        order = {_TN: 0, _FP: 1, _AMB: 2, _FN: 3, _TP: 4}
        stats["_order"] = stats["error_type"].map(order)
        return stats.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Importance des features sur les erreurs
    # ------------------------------------------------------------------

    def feature_importance_on_errors(
        self,
        analyzed:      pd.DataFrame,
        feature_cols:  Optional[list] = None,
        top_n:         int = 15,
    ) -> pd.DataFrame:
        """
        Compare les valeurs moyennes des features entre les erreurs et les
        prédictions correctes pour identifier les features discriminantes.

        Méthode : différence standardisée (effect size Cohen's d simplifié)
        entre les FP/FN et les prédictions correctes.

        Args:
            analyzed      : DataFrame annoté par analyze()
            feature_cols  : colonnes de features à analyser (auto-detect si None)
            top_n         : nb de features à retourner

        Returns:
            DataFrame trié par |effect_size| décroissant
        """
        if feature_cols is None:
            # Exclure les colonnes de métadonnées
            meta_cols = {"account_id", "y_true", "y_prob", "y_pred",
                         "error_type", "confidence_level", "confidence_score",
                         "margin_from_threshold", "is_uncertain", "rank_in_category"}
            feature_cols = [c for c in analyzed.columns
                            if c not in meta_cols
                            and pd.api.types.is_numeric_dtype(analyzed[c])]

        if not feature_cols:
            return pd.DataFrame(columns=["feature", "effect_size", "mean_correct", "mean_error"])

        # Masques
        correct_mask = analyzed["error_type"].isin([_TP, _TN])
        error_mask   = analyzed["error_type"].isin([_FP, _FN, _AMB])

        rows = []
        for col in feature_cols:
            if col not in analyzed.columns:
                continue
            v_correct = analyzed.loc[correct_mask, col].dropna()
            v_error   = analyzed.loc[error_mask,   col].dropna()

            if len(v_correct) < 2 or len(v_error) < 2:
                continue

            mean_c = float(v_correct.mean())
            mean_e = float(v_error.mean())
            pooled_std = float(
                np.sqrt((v_correct.var() + v_error.var()) / 2 + 1e-9)
            )
            effect = (mean_e - mean_c) / pooled_std

            rows.append({
                "feature":      col,
                "effect_size":  round(effect, 4),
                "mean_correct": round(mean_c, 4),
                "mean_error":   round(mean_e, 4),
                "abs_effect":   round(abs(effect), 4),
            })

        if not rows:
            return pd.DataFrame()

        df_fi = (pd.DataFrame(rows)
                 .sort_values("abs_effect", ascending=False)
                 .drop(columns="abs_effect")
                 .head(top_n)
                 .reset_index(drop=True))
        return df_fi

    # ------------------------------------------------------------------
    # Rapport textuel synthétique
    # ------------------------------------------------------------------

    def report(self, analyzed: pd.DataFrame) -> str:
        """
        Génère un rapport textuel complet, prêt à afficher dans le terminal.
        """
        n   = len(analyzed)
        err = analyzed["error_type"].value_counts()
        n_fp  = int(err.get(_FP,  0))
        n_fn  = int(err.get(_FN,  0))
        n_tp  = int(err.get(_TP,  0))
        n_tn  = int(err.get(_TN,  0))
        n_amb = int(err.get(_AMB, 0))

        accuracy = round((n_tp + n_tn) / max(n, 1) * 100, 2)
        fp_rate  = round(n_fp / max(n_tn + n_fp, 1) * 100, 2)
        fn_rate  = round(n_fn / max(n_tp + n_fn, 1) * 100, 2)

        lines = [
            f"\n{'='*60}",
            f"  Rapport d'analyse d'erreurs",
            f"  Seuil={self.threshold}  Zone grise=[{self.low_bound:.2f},{self.high_bound:.2f}]",
            f"{'='*60}",
            f"  {'Total':<22} {n:>6}",
            f"  {'Vrais positifs (TP)':<22} {n_tp:>6}",
            f"  {'Vrais négatifs (TN)':<22} {n_tn:>6}",
            f"  {'Faux positifs (FP)':<22} {n_fp:>6}  ({fp_rate:.1f}% des humains)",
            f"  {'Faux négatifs (FN)':<22} {n_fn:>6}  ({fn_rate:.1f}% des bots)",
            f"  {'Ambigus':<22} {n_amb:>6}",
            f"  {'-'*40}",
            f"  Accuracy                {accuracy:.2f}%",
            f"{'='*60}",
        ]

        # Score distribution compacte
        dist = self.score_distribution(analyzed)
        lines.append("\n  Distribution des scores par catégorie :")
        lines.append(f"  {'Type':<18} {'N':>5} {'mean':>7} {'std':>7} {'median':>7}")
        lines.append(f"  {'-'*46}")
        for _, row in dist.iterrows():
            lines.append(
                f"  {row['error_type']:<18} {int(row['count']):>5} "
                f"{row['mean']:>7.4f} {row['std']:>7.4f} {row['median']:>7.4f}"
            )

        lines.append(f"\n{'='*60}")

        # Top FP et FN
        fp_df = self.false_positives(analyzed, top_n=5)
        fn_df = self.false_negatives(analyzed, top_n=5)

        if not fp_df.empty:
            lines.append(f"\n  Top 5 Faux Positifs (humains classés bots) :")
            for _, row in fp_df.iterrows():
                lines.append(f"    {row['account_id']}  prob={row['y_prob']:.4f}")

        if not fn_df.empty:
            lines.append(f"\n  Top 5 Faux Négatifs (bots non détectés) :")
            for _, row in fn_df.iterrows():
                lines.append(f"    {row['account_id']}  prob={row['y_prob']:.4f}")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)
