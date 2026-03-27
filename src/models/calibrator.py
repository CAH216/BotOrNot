# -*- coding: utf-8 -*-
"""
src/models/calibrator.py
-------------------------
Calibration des probabilités pour BotOrNot.

Problème : les modèles ensemblistes (LightGBM, XGBoost) produisent souvent
des probabilités "écrasées" vers 0 ou 1, ce qui rend le choix du seuil
peu fiable. La calibration corrige cette distorsion.

Méthodes disponibles :
    - Platt scaling  : régression logistique sur les scores bruts (rapide,
                       fiable si le dataset de calibration est petit)
    - Isotonic       : régression isotonique non-paramétrique (plus puissante
                       mais nécessite plus de données ~1000 exemples)

Usage :
    from src.models.calibrator import ProbabilityCalibrator

    cal = ProbabilityCalibrator(method="platt")
    cal.fit(model, X_val, y_val)         # calibrer sur la validation
    proba_cal = cal.predict_proba(X_test)
    print(cal.calibration_report(X_val, y_val))
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from src.models._base import BotDetectorBase, compute_metrics

logger = logging.getLogger(__name__)

CalibrationMethod = Literal["platt", "isotonic"]


# ---------------------------------------------------------------------------
# Utilitaires : mesure de calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calcule l'Expected Calibration Error (ECE).

    ECE ≈ 0 → probabilités parfaitement calibrées
    ECE ≈ 0.1 → erreur moyenne de 10 points de pourcentage

    Args:
        y_true : labels vrais (0/1)
        y_prob : probabilités prédites
        n_bins : nb de bins

    Returns:
        ECE (float, entre 0 et 1)
    """
    bins = np.linspace(0., 1., n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(y_prob[mask].mean())
        bin_acc  = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return round(ece, 4)


def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Retourne les données de la courbe de calibration (reliability diagram).

    Returns:
        DataFrame avec colonnes : bin_center, mean_predicted, fraction_positives, count
    """
    bins       = np.linspace(0., 1., n_bins + 1)
    rows       = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        count  = mask.sum()
        if count == 0:
            continue
        rows.append({
            "bin_center":         round((lo + hi) / 2, 3),
            "mean_predicted":     round(float(y_prob[mask].mean()), 4),
            "fraction_positives": round(float(y_true[mask].mean()), 4),
            "count":              int(count),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ProbabilityCalibrator
# ---------------------------------------------------------------------------

class ProbabilityCalibrator:
    """
    Calibrateur de probabilités — wrapper autour d'un BotDetectorBase.

    Le calibrateur enveloppe un modèle entraîné et corrige ses probabilités
    de sortie via Platt scaling ou régression isotonique.

    Flux :
        1. Entraîner le modèle de base sur train
        2. Obtenir les scores sur val (données non vues)
        3. Calibrer sur val : cal.fit(model, X_val, y_val)
        4. À l'inférence : cal.predict_proba(X_test)

    IMPORTANT : ne jamais calibrer sur le jeu d'entraînement
    (sur-apprentissage garanti).
    """

    def __init__(
        self,
        method:  CalibrationMethod = "platt",
        n_bins:  int = 10,
    ) -> None:
        self.method     = method
        self.n_bins     = n_bins
        self.model_:    Optional[BotDetectorBase] = None
        self.calibrator_ = None
        self.is_fitted_  = False
        self.ece_before_ = None
        self.ece_after_  = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        model:  BotDetectorBase,
        X_cal:  pd.DataFrame,
        y_cal:  pd.Series,
    ) -> "ProbabilityCalibrator":
        """
        Calibre le modèle sur un jeu de calibration (validation recommandé).

        Args:
            model : modèle de base déjà entraîné
            X_cal : features de calibration (DIFFÉRENT du jeu d'entraînement)
            y_cal : labels de calibration

        Returns:
            self
        """
        self.model_ = model

        # Scores bruts du modèle de base
        raw_scores = model.predict_proba(X_cal)
        y_arr      = np.array(y_cal, dtype=int)

        # ECE avant calibration
        self.ece_before_ = expected_calibration_error(y_arr, raw_scores, self.n_bins)
        logger.info("[Calibrator] ECE avant = %.4f (méthode=%s)", self.ece_before_, self.method)

        # Entraîner le calibrateur
        if self.method == "platt":
            self.calibrator_ = self._fit_platt(raw_scores, y_arr)
        elif self.method == "isotonic":
            self.calibrator_ = self._fit_isotonic(raw_scores, y_arr)
        else:
            raise ValueError(f"Méthode inconnue : '{self.method}'")

        # ECE après calibration
        cal_scores       = self._apply_calibrator(raw_scores)
        self.ece_after_  = expected_calibration_error(y_arr, cal_scores, self.n_bins)
        logger.info("[Calibrator] ECE après = %.4f (amélioration=%.4f)",
                    self.ece_after_, self.ece_before_ - self.ece_after_)

        self.is_fitted_ = True
        return self

    @staticmethod
    def _fit_platt(scores: np.ndarray, y: np.ndarray):
        """
        Platt scaling : LR logistique sur les scores bruts.
        Transforme scores → probabilités calibrées via σ(a*score + b).
        """
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(scores.reshape(-1, 1), y)
        return lr

    @staticmethod
    def _fit_isotonic(scores: np.ndarray, y: np.ndarray):
        """
        Régression isotonique : courbe monotone non-paramétrique.
        Plus puissante que Platt mais nécessite davantage de données.
        """
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores, y)
        return iso

    def _apply_calibrator(self, scores: np.ndarray) -> np.ndarray:
        """Applique la transformation de calibration sur des scores bruts."""
        if self.method == "platt":
            return self.calibrator_.predict_proba(scores.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            return self.calibrator_.predict(scores)
        return scores

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retourne les probabilités calibrées pour X.

        Args:
            X : matrice de features

        Returns:
            array de probabilités calibrées (0.0–1.0)
        """
        self._check_fitted()
        raw_scores = self.model_.predict_proba(X)
        return self._apply_calibrator(raw_scores)

    def predict(
        self,
        X:         pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Retourne les labels binaires avec probabilités calibrées."""
        thr   = threshold or self.model_.threshold or 0.5
        proba = self.predict_proba(X)
        return (proba >= thr).astype(int)

    # ------------------------------------------------------------------
    # Rapport de calibration
    # ------------------------------------------------------------------

    def calibration_report(
        self,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        metrics: Optional[list] = None,
    ) -> str:
        """
        Rapport de calibration textuel : ECE avant/après + métriques.

        Args:
            X_eval  : features d'évaluation
            y_eval  : labels vrais

        Returns:
            Rapport formaté en texte
        """
        self._check_fitted()
        from src.evaluation.metrics import DEFAULT_METRICS

        y_arr      = np.array(y_eval, dtype=int)
        raw_scores = self.model_.predict_proba(X_eval)
        cal_scores = self._apply_calibrator(raw_scores)

        ece_before = expected_calibration_error(y_arr, raw_scores, self.n_bins)
        ece_after  = expected_calibration_error(y_arr, cal_scores, self.n_bins)

        used_metrics = metrics or ["roc_auc", "f1", "pr_auc"]

        from src.evaluation.metrics import compute_metrics as _cm
        m_before = _cm(y_arr, raw_scores, metrics=used_metrics)
        m_after  = _cm(y_arr, cal_scores, metrics=used_metrics)

        lines = [
            f"\n{'='*55}",
            f"  Rapport calibration [{self.model_.name}] — {self.method}",
            f"{'='*55}",
            f"  {'Métrique':<24} {'Avant':>8} {'Après':>8}",
            f"  {'-'*42}",
            f"  {'ECE':<24} {ece_before:>8.4f} {ece_after:>8.4f}  {'↑' if ece_after < ece_before else '↓'}",
        ]

        for m in used_metrics:
            before = m_before.get(m, float("nan"))
            after  = m_after.get(m, float("nan"))
            arrow  = "↑" if after > before else ("↓" if after < before else "=")
            lines.append(f"  {m:<24} {before:>8.4f} {after:>8.4f}  {arrow}")

        improvement = ece_before - ece_after
        lines += [
            f"{'='*55}",
            f"  ECE améliorée de {improvement:.4f} ({improvement/max(ece_before,1e-9)*100:.1f}%)",
            f"{'='*55}",
        ]
        return "\n".join(lines)

    def compare_curves(
        self,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame comparant les courbes de calibration
        avant et après correction.

        Returns:
            DataFrame avec colonnes : bin_center, mean_pred_raw, frac_pos_raw,
                                      mean_pred_cal, frac_pos_cal
        """
        self._check_fitted()
        y_arr      = np.array(y_eval, dtype=int)
        raw_scores = self.model_.predict_proba(X_eval)
        cal_scores = self._apply_calibrator(raw_scores)

        curve_raw = calibration_curve_data(y_arr, raw_scores, self.n_bins)
        curve_cal = calibration_curve_data(y_arr, cal_scores, self.n_bins)

        merged = curve_raw.rename(columns={
            "mean_predicted":     "mean_pred_raw",
            "fraction_positives": "frac_pos_raw",
        }).merge(
            curve_cal.rename(columns={
                "mean_predicted":     "mean_pred_cal",
                "fraction_positives": "frac_pos_cal",
            }),
            on="bin_center",
            how="outer",
        ).sort_values("bin_center").reset_index(drop=True)

        return merged

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.calibrator_ is None:
            raise RuntimeError(
                "ProbabilityCalibrator non calibré. Appeler fit() d'abord."
            )
