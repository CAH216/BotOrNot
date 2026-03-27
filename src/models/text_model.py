# -*- coding: utf-8 -*-
"""
src/models/text_model.py
-------------------------
Modèle texte indépendant — TF-IDF au niveau compte + classifieur léger.

Pourquoi un modèle texte séparé ?
    Le bloc tabular/temporal capture les *patterns comportementaux*.
    Ce module capture les *patterns de contenu* : un bot peut avoir des
    timestamps plausibles mais un texte généré ou copié-collé.
    La fusion des deux scores améliore la robustesse selon l'architecture V1.

Pipeline :
    1. Agréger tous les posts d'un compte en un seul "document"
    2. Vectoriser via TF-IDF (sur les mots, ou sur les char n-grams)
    3. Entraîner un classifieur rapide (LR ou SGD)
    4. Sortie = prob_bot_text par compte

Sortie :
    DataFrame avec account_id + prob_bot_text
    Compatible avec FeatureAssembler via block "text_embeddings"

Usage :
    from src.models.text_model import TextBotDetector

    model = TextBotDetector()
    model.fit(posts_train, labels_train)
    proba_df = model.predict_proba_df(posts_test)   # account_id + prob_bot_text
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.data.schema import AccountCols, PostCols, LabelCols

logger = logging.getLogger(__name__)

Vectorizer = Literal["word", "char", "char_wb"]


# ---------------------------------------------------------------------------
# TextBotDetector
# ---------------------------------------------------------------------------

class TextBotDetector:
    """
    Détecteur basé sur le contenu textuel agrégé par compte.

    Args:
        vectorizer     : "word" (mots) | "char" (char n-grams) | "char_wb" (word-boundary)
        ngram_range    : plage de n-grammes (ex: (1,2) = uni+bigrammes)
        max_features   : nb max de features TF-IDF
        classifier     : "lr" (Logistic Regression, stable) | "sgd" (SGD, rapide)
        min_docs       : nb minimum de posts par compte pour inclure dans le train
        text_col       : colonne texte à utiliser
    """

    name = "text_tfidf"

    def __init__(
        self,
        vectorizer:   Vectorizer = "word",
        ngram_range:  tuple = (1, 2),
        max_features: int   = 10_000,
        classifier:   Literal["lr", "sgd"] = "lr",
        min_docs:     int   = 1,
        text_col:     Optional[str] = None,
        random_state: int   = 42,
    ) -> None:
        self.vectorizer   = vectorizer
        self.ngram_range  = ngram_range
        self.max_features = max_features
        self.classifier   = classifier
        self.min_docs     = min_docs
        self.text_col     = text_col
        self.random_state = random_state

        self.vectorizer_  = None
        self.classifier_  = None
        self.is_fitted_   = False
        self.threshold_   = 0.5

    # ------------------------------------------------------------------
    # Agrégation texte : posts → document par compte
    # ------------------------------------------------------------------

    def _aggregate_texts(
        self,
        posts_df:  pd.DataFrame,
        text_col:  str,
    ) -> pd.Series:
        """
        Agrège tous les posts d'un compte en un seul document.

        Stratégie : concaténation avec espace — simple et efficace pour TF-IDF.
        Le TF-IDF pondère naturellement les termes fréquents dans le compte.

        Returns:
            Series indexée par account_id, valeur = texte agrégé
        """
        id_col = AccountCols.ID
        return (
            posts_df
            .groupby(id_col)[text_col]
            .agg(lambda texts: " ".join(texts.fillna("").astype(str)))
        )

    def _choose_text_col(self, posts_df: pd.DataFrame) -> str:
        """Choisit la meilleure colonne texte disponible."""
        if self.text_col and self.text_col in posts_df.columns:
            return self.text_col
        # Priorité : text_clean > text
        for col in [PostCols.TEXT_CLEAN, PostCols.TEXT, "text_clean", "text"]:
            if col in posts_df.columns:
                return col
        raise ValueError(
            "Aucune colonne texte trouvée dans posts_df. "
            f"Colonnes disponibles : {list(posts_df.columns)}"
        )

    # ------------------------------------------------------------------
    # Build du pipeline sklearn
    # ------------------------------------------------------------------

    def _build_pipeline(self):
        """Construit le pipeline TF-IDF + classifieur."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.pipeline import Pipeline

        # Vectoriseur
        analyzer = {
            "word":    "word",
            "char":    "char",
            "char_wb": "char_wb",
        }.get(self.vectorizer, "word")

        tfidf = TfidfVectorizer(
            analyzer     = analyzer,
            ngram_range  = self.ngram_range,
            max_features = self.max_features,
            sublinear_tf = True,        # log(1 + tf) → réduit l'effet des mots très fréquents
            strip_accents = "unicode",
            min_df       = 2,           # ignorer les termes hapax
        )

        # Classifieur
        if self.classifier == "sgd":
            clf = SGDClassifier(
                loss         = "modified_huber",  # produit des probabilités
                class_weight = "balanced",
                random_state = self.random_state,
                max_iter     = 200,
                n_jobs       = -1,
            )
        else:
            clf = LogisticRegression(
                C            = 1.0,
                class_weight = "balanced",
                solver       = "lbfgs",
                max_iter     = 500,
                random_state = self.random_state,
            )

        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def fit(
        self,
        posts_df:  pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> "TextBotDetector":
        """
        Entraîne le modèle texte.

        Args:
            posts_df  : DataFrame de posts avec account_id + texte
            labels_df : DataFrame avec account_id + label (0/1)

        Returns:
            self
        """
        id_col  = AccountCols.ID
        text_col = self._choose_text_col(posts_df)

        if id_col not in labels_df.columns or LabelCols.LABEL not in labels_df.columns:
            raise ValueError(f"labels_df doit contenir '{id_col}' et '{LabelCols.LABEL}'")

        # Agréger les posts par compte
        docs = self._aggregate_texts(posts_df, text_col)

        # Filtrer les comptes avec assez de posts
        if self.min_docs > 1:
            n_posts = posts_df.groupby(id_col).size()
            valid   = n_posts[n_posts >= self.min_docs].index
            docs    = docs[docs.index.isin(valid)]

        # Aligner avec les labels
        label_map = labels_df.set_index(id_col)[LabelCols.LABEL]
        common    = docs.index.intersection(label_map.index)

        if len(common) == 0:
            raise ValueError("Aucun compte commun entre posts_df et labels_df")

        X_text = docs.loc[common].values.tolist()
        y      = label_map.loc[common].values.astype(int)

        logger.info(
            "[%s] Entraînement : %d comptes (pos=%.1f%%)",
            self.name, len(X_text), 100 * y.mean()
        )

        self.vectorizer_ = self._build_pipeline()
        self.vectorizer_.fit(X_text, y)
        self.is_fitted_ = True

        return self

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict_proba_df(
        self,
        posts_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame avec account_id + prob_bot_text.

        Compatible avec FeatureAssembler (block name: "text_embeddings").

        Args:
            posts_df : DataFrame de posts (même format qu'à l'entraînement)

        Returns:
            DataFrame index=account_id, colonnes=[account_id, prob_bot_text]
        """
        self._check_fitted()
        id_col   = AccountCols.ID
        text_col = self._choose_text_col(posts_df)
        docs     = self._aggregate_texts(posts_df, text_col)

        X_text   = docs.values.tolist()
        proba    = self.vectorizer_.predict_proba(X_text)[:, 1]

        return pd.DataFrame({
            id_col:         docs.index.tolist(),
            "prob_bot_text": np.round(proba, 4),
        }).reset_index(drop=True)

    def predict_proba(self, posts_df: pd.DataFrame) -> np.ndarray:
        """Retourne uniquement le vecteur de probabilités (sans les IDs)."""
        df = self.predict_proba_df(posts_df)
        return df["prob_bot_text"].values

    def predict(
        self,
        posts_df:  pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Retourne les labels binaires."""
        thr   = threshold or self.threshold_
        proba = self.predict_proba(posts_df)
        return (proba >= thr).astype(int)

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """
        Retourne les features TF-IDF les plus importantes pour la classe bot.

        Returns:
            DataFrame avec colonnes : feature, coef_bot
            Trié par coefficient décroissant.
        """
        self._check_fitted()
        tfidf = self.vectorizer_.named_steps["tfidf"]
        clf   = self.vectorizer_.named_steps["clf"]

        if not hasattr(clf, "coef_"):
            logger.warning("Le classifieur ne supporte pas coef_")
            return pd.DataFrame()

        feature_names = tfidf.get_feature_names_out()
        coef          = clf.coef_[0]

        return (
            pd.DataFrame({"feature": feature_names, "coef_bot": coef})
            .sort_values("coef_bot", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Sauvegarde le pipeline en joblib + métadonnées JSON."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer_, path.with_suffix(".joblib"))
        meta = {
            "model":       self.name,
            "vectorizer":  self.vectorizer,
            "ngram_range": list(self.ngram_range),
            "max_features":self.max_features,
            "classifier":  self.classifier,
            "threshold":   self.threshold_,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("[%s] Sauvegardé : %s", self.name, path)

    @classmethod
    def load(cls, path: str | Path) -> "TextBotDetector":
        """Charge un modèle sauvegardé."""
        import joblib
        path = Path(path)
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        instance = cls(
            vectorizer   = meta.get("vectorizer", "word"),
            ngram_range  = tuple(meta.get("ngram_range", [1, 2])),
            max_features = meta.get("max_features", 10000),
            classifier   = meta.get("classifier", "lr"),
        )
        instance.vectorizer_  = joblib.load(path.with_suffix(".joblib"))
        instance.threshold_   = meta.get("threshold", 0.5)
        instance.is_fitted_   = True
        return instance

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.vectorizer_ is None:
            raise RuntimeError(
                f"[{self.name}] Non entraîné. Appeler fit() d'abord."
            )
