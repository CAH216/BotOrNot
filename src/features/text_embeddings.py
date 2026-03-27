# -*- coding: utf-8 -*-
"""
src/features/text_embeddings.py
--------------------------------
Couche NLP avancée — embeddings de phrase optionnels.

Ce module est OPTIONNEL et CONDITIONNEL :
    - Il ne s'active que si `sentence-transformers` est installé
    - Il ne s'active que si le profiler détecte assez de texte (has_text=True)
    - Il produit des features denses utiles pour les cas où le TF-IDF sature

Pourquoi des embeddings ?
    Le TF-IDF ne capture pas la sémantique. Deux phrases identiques dans le
    sens mais écrites différemment ont un TF-IDF très différent.
    Les embeddings capturent le sens → détection de paraphrase de bot.

Modèles légers recommandés (CPU-friendly) :
    - "all-MiniLM-L6-v2"    : 22M params, rapide, excellent général (~80MB)
    - "paraphrase-MiniLM-L3-v2" : encore plus léger (~60MB)
    - "multilingual-e5-small"   : multilingue (~100MB)

Pipeline par compte :
    1. Agréger les posts (jusqu'à max_posts_per_account)
    2. Encoder chaque post en vecteur 384-dim (ou 768)
    3. Agréger les vecteurs (mean + max pooling)
    4. Calculer des features de variance/cohérence inter-posts
    5. Retourner un DF de features par compte

Usage :
    from src.features.text_embeddings import TextEmbeddingExtractor

    ext = TextEmbeddingExtractor(model_name="all-MiniLM-L6-v2")
    if ext.is_available():
        feat_df = ext.extract(posts_df)
    else:
        feat_df = pd.DataFrame()   # pipeline continue sans lui
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import AccountCols, PostCols

logger = logging.getLogger(__name__)

# Modèle par défaut (léger, CPU-friendly)
_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Nombre max de posts encodés par compte (performance vs précision)
_MAX_POSTS_PER_ACCOUNT = 30

# Préfixe des features produites
_PREFIX = "emb_"


# ---------------------------------------------------------------------------
# Vérification de disponibilité
# ---------------------------------------------------------------------------

def _sentence_transformers_available() -> bool:
    """Vérifie si sentence-transformers est installé sans crasher."""
    try:
        import importlib
        importlib.import_module("sentence_transformers")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TextEmbeddingExtractor
# ---------------------------------------------------------------------------

class TextEmbeddingExtractor:
    """
    Extracteur de features basé sur des embeddings de phrase.

    Toujours s'assurer que is_available() est True avant d'appeler extract().

    Args:
        model_name          : nom du modèle HuggingFace/SentenceTransformers
        max_posts_per_account : nb max de posts encodés par compte
        batch_size          : taille des batchs d'encodage
        device              : "cpu" | "cuda" | None (auto-detect)
        cache_folder        : dossier de cache du modèle
    """

    def __init__(
        self,
        model_name:            str = _DEFAULT_MODEL,
        max_posts_per_account: int = _MAX_POSTS_PER_ACCOUNT,
        batch_size:            int = 64,
        device:                Optional[str] = None,
        cache_folder:          Optional[str] = None,
    ) -> None:
        self.model_name            = model_name
        self.max_posts_per_account = max_posts_per_account
        self.batch_size            = batch_size
        self.device                = device
        self.cache_folder          = cache_folder
        self._model                = None   # chargé lazily

    # ------------------------------------------------------------------
    # Disponibilité
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Retourne True si sentence-transformers est installé."""
        return _sentence_transformers_available()

    def _load_model(self):
        """Charge le modèle lazily (évite le coût au import)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device        = self.device,
                cache_folder  = self.cache_folder,
            )
            logger.info(
                "[embeddings] Modèle chargé : %s (dim=%d)",
                self.model_name, self._model.get_sentence_embedding_dimension()
            )
        return self._model

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        posts_df:  pd.DataFrame,
        text_col:  Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extrait des features d'embedding par compte.

        Features produites (préfixe `emb_`) :
            emb_mean_*      : moyenne des vecteurs d'embedding par compte
            emb_max_*       : max pooling des vecteurs
            emb_var_mean    : variance moyenne des embeddings (cohérence)
            emb_cosim_mean  : similarité cosinus moyenne entre posts
            emb_cosim_std   : écart-type des similarités

        Args:
            posts_df : DataFrame avec account_id + colonne texte
            text_col : colonne texte à utiliser (auto-détect si None)

        Returns:
            DataFrame avec account_id + features emb_*
            Vide si extraction impossible.
        """
        if not self.is_available():
            logger.warning(
                "[embeddings] sentence-transformers non installé. "
                "Lancer : pip install sentence-transformers"
            )
            return pd.DataFrame(columns=[AccountCols.ID])

        id_col   = AccountCols.ID
        text_col = self._detect_text_col(posts_df, text_col)
        if text_col is None or id_col not in posts_df.columns:
            return pd.DataFrame(columns=[id_col])

        model = self._load_model()
        all_rows = []

        accounts = posts_df[id_col].unique()
        logger.info("[embeddings] Encodage de %d comptes...", len(accounts))

        for account_id in accounts:
            group      = posts_df[posts_df[id_col] == account_id]
            texts      = group[text_col].dropna().astype(str).tolist()
            # Limiter pour la performance
            texts      = texts[:self.max_posts_per_account]

            if not texts:
                all_rows.append({id_col: account_id, **self._null_features(model)})
                continue

            try:
                embeddings = model.encode(
                    texts,
                    batch_size     = self.batch_size,
                    show_progress_bar = False,
                    convert_to_numpy  = True,
                )   # shape: (n_posts, dim)

                feat = self._aggregate_embeddings(embeddings)
                feat[id_col] = account_id
                all_rows.append(feat)

            except Exception as e:
                logger.warning("[embeddings] Erreur compte '%s' : %s", account_id, e)
                all_rows.append({id_col: account_id, **self._null_features(model)})

        if not all_rows:
            return pd.DataFrame(columns=[id_col])

        result = pd.DataFrame(all_rows)
        # Mettre account_id en premier
        cols = [id_col] + [c for c in result.columns if c != id_col]
        result = result[cols].reset_index(drop=True)

        logger.info(
            "[embeddings] %d comptes × %d features",
            len(result), len(result.columns) - 1
        )
        return result

    # ------------------------------------------------------------------
    # Agrégation des embeddings
    # ------------------------------------------------------------------

    def _aggregate_embeddings(self, embeddings: np.ndarray) -> dict:
        """
        Agrège une matrice d'embeddings (n_posts × dim) en features scalaires.

        Features :
            - mean pooling (dim valeurs)
            - max pooling  (dim valeurs)
            - variance moyenne → cohérence sémantique du compte
            - similarité cosinus inter-posts → détection de copier-coller sémantique
        """
        n, dim = embeddings.shape
        feat:  dict = {}

        # Mean & Max pooling (features denses)
        mean_emb = embeddings.mean(axis=0)
        max_emb  = embeddings.max(axis=0)

        for i, v in enumerate(mean_emb):
            feat[f"{_PREFIX}mean_{i}"] = round(float(v), 6)
        for i, v in enumerate(max_emb):
            feat[f"{_PREFIX}max_{i}"] = round(float(v), 6)

        # Variance moyenne (cohérence) : bots → très faible (posts identiques)
        feat[f"{_PREFIX}var_mean"] = round(float(embeddings.var(axis=0).mean()), 6)
        feat[f"{_PREFIX}var_std"]  = round(float(embeddings.var(axis=0).std()), 6)

        # Similarité cosinus inter-posts
        if n > 1:
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1e-9, norm)
            normed = embeddings / norm
            sim_matrix = normed @ normed.T   # (n × n)
            # Prendre triangulaire supérieure (sans diagonale)
            idx = np.triu_indices(n, k=1)
            sims = sim_matrix[idx]
            feat[f"{_PREFIX}cosim_mean"] = round(float(sims.mean()), 6)
            feat[f"{_PREFIX}cosim_std"]  = round(float(sims.std()),  6)
            feat[f"{_PREFIX}cosim_max"]  = round(float(sims.max()),  6)
        else:
            feat[f"{_PREFIX}cosim_mean"] = float("nan")
            feat[f"{_PREFIX}cosim_std"]  = float("nan")
            feat[f"{_PREFIX}cosim_max"]  = float("nan")

        return feat

    def _null_features(self, model) -> dict:
        """Retourne un dict de NaN pour les comptes sans texte."""
        dim = model.get_sentence_embedding_dimension()
        feat: dict = {}
        for i in range(dim):
            feat[f"{_PREFIX}mean_{i}"] = float("nan")
            feat[f"{_PREFIX}max_{i}"]  = float("nan")
        for key in ["var_mean", "var_std", "cosim_mean", "cosim_std", "cosim_max"]:
            feat[f"{_PREFIX}{key}"] = float("nan")
        return feat

    @staticmethod
    def _detect_text_col(posts_df: pd.DataFrame, text_col: Optional[str]) -> Optional[str]:
        """Auto-détecte la colonne texte."""
        if text_col and text_col in posts_df.columns:
            return text_col
        for col in [PostCols.TEXT_CLEAN, PostCols.TEXT, "text_clean", "text", "content"]:
            if col in posts_df.columns:
                return col
        logger.warning("[embeddings] Aucune colonne texte trouvée")
        return None


# ---------------------------------------------------------------------------
# Fonction de convenance pour intégration pipeline
# ---------------------------------------------------------------------------

def extract_text_embeddings(
    posts_df:   pd.DataFrame,
    model_name: str = _DEFAULT_MODEL,
    text_col:   Optional[str] = None,
    max_posts:  int = _MAX_POSTS_PER_ACCOUNT,
    device:     Optional[str] = None,
) -> pd.DataFrame:
    """
    Fonction de convenance pour l'assembleur de features.

    Si sentence-transformers n'est pas installé → DataFrame vide propre.
    Si installé → retourne les features emb_* par compte.

    Args:
        posts_df   : DataFrame de posts
        model_name : modèle à utiliser
        text_col   : colonne texte
        max_posts  : nb max de posts par compte
        device     : "cpu" ou "cuda"

    Returns:
        DataFrame avec account_id + features emb_*,
        OU DataFrame vide si non disponible.
    """
    ext = TextEmbeddingExtractor(
        model_name            = model_name,
        max_posts_per_account = max_posts,
        device                = device,
    )

    if not ext.is_available():
        logger.info(
            "[embeddings] Module désactivé — sentence-transformers absent. "
            "Installation : pip install sentence-transformers"
        )
        return pd.DataFrame(columns=[AccountCols.ID])

    return ext.extract(posts_df, text_col=text_col)
