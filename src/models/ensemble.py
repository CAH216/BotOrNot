# -*- coding: utf-8 -*-
"""
src/models/ensemble.py
-----------------------
Ensemble final — fusion pondérée de plusieurs scores de sous-modèles.

Design :
    Chaque sous-modèle produit une probabilité independante (0.0–1.0).
    L'EnsembleBlender fusionne ces probabilités avec des poids configurables
    et gère proprement les modules absents (NaN → ignoré automatiquement).

Philosophie :
    - Simple >> complexe (pas de stacking, pas de meta-learner au niveau 1)
    - Robuste aux modules manquants : un module absent n'invalide pas l'ensemble
    - Poids configurables : réglables selon la performance en validation
    - Traçable : le rapport indique quels modules ont contribué à chaque compte

Méthodes supportées :
    weighted_mean  → moyenne pondérée (méthode principale)
    mean           → moyenne simple (poids identiques)
    max            → proba max parmi les modules (approche maximisante)
    rank_mean      → moyenne des rangs (moins sensible aux valeurs aberrantes)

Usage :
    from src.models.ensemble import EnsembleBlender, EnsembleConfig

    config = EnsembleConfig(
        weights = {"tabular": 1.5, "temporal": 1.0, "text": 0.8},
        method  = "weighted_mean",
        threshold = 0.5,
    )
    blender = EnsembleBlender(config)

    # Fusionner des DataFrames de probabilités (account_id + prob_*)
    result = blender.blend([tabular_proba_df, temporal_proba_df, text_proba_df])
    print(result.head())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BlendMethod = Literal["weighted_mean", "mean", "max", "rank_mean"]


# ---------------------------------------------------------------------------
# EnsembleConfig — configuration centralisée
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """
    Configuration de l'ensemble.

    Args:
        weights   : dict {nom_module → poids} (normalisés automatiquement)
                    Si un module est absent, il est ignoré.
        method    : méthode de fusion
        threshold : seuil de décision final
        min_modules : nb minimum de modules requis (sinon → NaN / défaut humain)
        default_absent : probabilité assignée à un module absent
                         None = ignorer le module absent (recommandé)
    """
    weights:        Dict[str, float] = field(default_factory=lambda: {
        "tabular":  1.5,
        "temporal": 1.2,
        "text":     0.8,
        "structural": 0.5,
        "relational": 0.6,
    })
    method:         BlendMethod = "weighted_mean"
    threshold:      float = 0.5
    min_modules:    int   = 1        # au moins 1 module doit être présent
    default_absent: Optional[float] = None   # None = exclure les NaN


# ---------------------------------------------------------------------------
# EnsembleBlender
# ---------------------------------------------------------------------------

class EnsembleBlender:
    """
    Fusionneur de probabilités multi-modèles.

    Accepte en entrée des DataFrames de la forme :
        account_id | prob_*   (une colonne de probabilité)
    ou directement des vecteurs numpy indexés sur les mêmes account_ids.

    Retourne un DataFrame :
        account_id | prob_ensemble | label | modules_used | n_modules_active
    """

    ID_COL = "account_id"

    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        self.config = config or EnsembleConfig()

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def blend(
        self,
        proba_dfs:    List[pd.DataFrame],
        module_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fusionne plusieurs DataFrames de probabilités.

        Args:
            proba_dfs    : liste de DataFrames avec account_id + colonne prob_*
                           OU une seule colonne numérique (index = account_id)
            module_names : noms des modules dans l'ordre de proba_dfs.
                           Si None → déduit depuis le nom de la colonne prob_*.

        Returns:
            DataFrame avec :
                account_id, prob_ensemble, label,
                modules_used (str), n_modules_active (int)
        """
        if not proba_dfs:
            raise ValueError("La liste proba_dfs est vide")

        # Normaliser en Series indexées par account_id
        series_list, names_list = self._normalize_inputs(proba_dfs, module_names)

        # Construire la matrice de probabilités
        all_ids = sorted(set().union(*[s.index for s in series_list]))
        prob_matrix = pd.DataFrame(index=all_ids)
        for name, s in zip(names_list, series_list):
            prob_matrix[name] = s.reindex(all_ids)

        # Compter les modules actifs par compte
        n_active = prob_matrix.notna().sum(axis=1)

        # Appliquer la valeur par défaut pour les absents si configuré
        if self.config.default_absent is not None:
            prob_matrix = prob_matrix.fillna(self.config.default_absent)

        # Fusion selon la méthode choisie
        final_prob = self._apply_method(prob_matrix, names_list)

        # Masquer les comptes sans assez de modules
        insufficient = n_active < self.config.min_modules
        if insufficient.any():
            logger.warning(
                "%d comptes ont moins de %d modules actifs → prob=NaN",
                insufficient.sum(), self.config.min_modules
            )
            final_prob[insufficient] = np.nan

        # Labels
        labels = (final_prob >= self.config.threshold).astype(int)
        labels[final_prob.isna()] = 0   # défaut humain si incertitude totale

        # Modules utilisés par compte
        modules_used = self._modules_used_str(prob_matrix, names_list)

        result = pd.DataFrame({
            self.ID_COL:          all_ids,
            "prob_ensemble":      np.round(final_prob.values, 4),
            "label":              labels.values,
            "n_modules_active":   n_active.values.astype(int),
            "modules_used":       modules_used,
        })

        n_bots = int((result["label"] == 1).sum())
        logger.info(
            "[Ensemble] %s | %d comptes → %d bots (%.1f%%) | seuil=%.2f",
            self.config.method, len(result),
            n_bots, 100 * n_bots / max(len(result), 1),
            self.config.threshold,
        )
        return result.reset_index(drop=True)

    def blend_arrays(
        self,
        account_ids:  pd.Series,
        proba_dict:   Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Variante directe : passe les probabilités comme dict {module → array}.

        Args:
            account_ids : Series de IDs (même ordre que les arrays)
            proba_dict  : {module_name → array de probabilités}

        Returns:
            Même format que blend()
        """
        dfs = []
        names = []
        for name, arr in proba_dict.items():
            s = pd.Series(
                arr, index=account_ids.values, name=f"prob_{name}"
            )
            dfs.append(s.reset_index().rename(columns={"index": self.ID_COL}))
            names.append(name)
        return self.blend(dfs, module_names=names)

    # ------------------------------------------------------------------
    # Méthodes de fusion
    # ------------------------------------------------------------------

    def _apply_method(
        self,
        prob_matrix: pd.DataFrame,
        names:       List[str],
    ) -> pd.Series:
        """Applique la méthode de fusion sur la matrice."""
        cfg = self.config

        if cfg.method == "mean":
            return prob_matrix.mean(axis=1, skipna=True)

        elif cfg.method == "weighted_mean":
            weights = np.array([
                cfg.weights.get(name, 1.0) for name in names
            ], dtype=float)
            # Calcul pondéré en ignorant les NaN
            weighted_sum = pd.Series(0.0, index=prob_matrix.index)
            weight_sum   = pd.Series(0.0, index=prob_matrix.index)
            for i, name in enumerate(names):
                col    = prob_matrix[name]
                mask   = col.notna()
                weighted_sum[mask] += col[mask] * weights[i]
                weight_sum[mask]   += weights[i]
            result = weighted_sum / weight_sum.replace(0, np.nan)
            return result

        elif cfg.method == "max":
            return prob_matrix.max(axis=1, skipna=True)

        elif cfg.method == "rank_mean":
            # Convertir chaque colonne en rangs (0–1), puis moyenner
            ranked = prob_matrix.rank(pct=True, na_option="keep")
            return ranked.mean(axis=1, skipna=True)

        else:
            raise ValueError(f"Méthode inconnue : '{cfg.method}'")

    # ------------------------------------------------------------------
    # Normalisation des entrées
    # ------------------------------------------------------------------

    def _normalize_inputs(
        self,
        proba_dfs:    List[pd.DataFrame],
        module_names: Optional[List[str]],
    ) -> tuple[List[pd.Series], List[str]]:
        """
        Normalise les DataFrames d'entrée en Series indexées par account_id.
        """
        series_list = []
        names_list  = []

        for i, df in enumerate(proba_dfs):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"proba_dfs[{i}] doit être un DataFrame")

            # Détecter la colonne de probabilité
            prob_cols = [c for c in df.columns if "prob" in c.lower()
                         and c != self.ID_COL]
            if not prob_cols:
                logger.warning("proba_dfs[%d] n'a pas de colonne 'prob_*' — ignoré", i)
                continue

            prob_col = prob_cols[0]

            # Nom du module
            if module_names and i < len(module_names):
                name = module_names[i]
            else:
                # Déduire depuis le nom : "prob_tabular" → "tabular"
                name = prob_col.replace("prob_", "").replace("_text", "text")

            # Construire la Series
            if self.ID_COL in df.columns:
                s = df.set_index(self.ID_COL)[prob_col]
            else:
                s = df[prob_col]

            series_list.append(s.astype(float))
            names_list.append(name)

        if not series_list:
            raise ValueError("Aucun DataFrame valide trouvé dans proba_dfs")

        return series_list, names_list

    @staticmethod
    def _modules_used_str(
        prob_matrix: pd.DataFrame,
        names:       List[str],
    ) -> List[str]:
        """Liste des modules actifs (non-NaN) par compte."""
        result = []
        for _, row in prob_matrix.iterrows():
            active = [n for n in names if pd.notna(row[n])]
            result.append("+".join(active) if active else "none")
        return result

    # ------------------------------------------------------------------
    # Rapport
    # ------------------------------------------------------------------

    def report(self, result: pd.DataFrame) -> str:
        """Rapport textuel de la fusion."""
        n = len(result)
        n_bots = int((result["label"] == 1).sum())
        lines = [
            f"\n{'='*55}",
            f"  Rapport Ensemble [{self.config.method}]",
            f"  Seuil={self.config.threshold}",
            f"{'='*55}",
            f"  Comptes total          {n:>6}",
            f"  Prédits bots           {n_bots:>6}  ({100*n_bots/max(n,1):.1f}%)",
            f"  Prédits humains        {n-n_bots:>6}  ({100*(n-n_bots)/max(n,1):.1f}%)",
        ]
        # Distribution nb de modules
        if "n_modules_active" in result.columns:
            for k, cnt in result["n_modules_active"].value_counts().sort_index().items():
                lines.append(f"  {k} module(s) actifs       {cnt:>6}")
        # Modules utilisés
        if "modules_used" in result.columns:
            lines.append("\n  Combinaisons de modules :")
            for combo, cnt in result["modules_used"].value_counts().head(5).items():
                lines.append(f"    {combo:<35} {cnt}")
        lines.append("=" * 55)
        return "\n".join(lines)
