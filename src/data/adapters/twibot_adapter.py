# -*- coding: utf-8 -*-
"""
src/data/adapters/twibot_adapter.py
-----------------------------------
Adaptateur spécifique pour le dataset massif TwiBot-22.
Ingère des dossiers complexes et retourne un DataBundle canonique
compréhensible par BotOrNot.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from src.data.adapters.base_adapter import BaseAdapter
from src.data.schema import DataBundle, AccountCols, PostCols, LabelCols

logger = logging.getLogger(__name__)


class TwiBot22Adapter(BaseAdapter):
    
    name = "twibot-22"
    description = "Adaptateur natif pour l'architecture TwiBot-22 (user.json, edge.csv...)."

    def load(
        self,
        base_path: Union[str, Path],
        nrows: Optional[int] = None,
        **kwargs
    ) -> DataBundle:
        """
        Déballe un dossier TwiBot-22 entier dans le DataBundle.
        """
        p = Path(base_path)
        if not p.is_dir():
            raise FileNotFoundError(f"TwiBot22Adapter attend un dossier, reçu : {p}")
            
        logger.info(f"[{self.name}] Ingestion du dataset TwiBot-22 depuis {p}")
        
        # 1. Labels
        label_df = None
        label_file = p / "label.csv"
        if label_file.exists():
            label_df = pd.read_csv(label_file)
            # TwiBot-22 label: id, label (bot/human)
            if "id" in label_df.columns:
                label_df.rename(columns={"id": LabelCols.ACCOUNT_ID}, inplace=True)
            if "label" in label_df.columns:
                label_df["label"] = label_df["label"].str.lower().map({"bot": 1, "human": 0}).astype(float)
                
        # 2. Utilisateurs
        accounts_df = None
        user_file = p / "user.json"
        if user_file.exists():
            accounts_df = pd.read_json(user_file, orient="records", nrows=nrows)
            # Mapping TwiBot-22 vers notre Schéma Canonique
            renames = {
                "id": AccountCols.ID,
                "name": AccountCols.SCREEN_NAME,
                "description": AccountCols.BIO,
                "created_at": AccountCols.CREATED_AT,
                "public_metrics.followers_count": AccountCols.FOLLOWERS,
                "public_metrics.following_count": AccountCols.FOLLOWING,
                "public_metrics.tweet_count": AccountCols.TOTAL_POSTS
            }
            # Unpack public_metrics s'il est present en dict (TwiBot-22 format)
            if "public_metrics" in accounts_df.columns and isinstance(accounts_df["public_metrics"].iloc[0], dict):
                metrics = accounts_df["public_metrics"].apply(pd.Series)
                accounts_df = pd.concat([accounts_df.drop("public_metrics", axis=1), metrics], axis=1)
                
            accounts_df.rename(columns=renames, inplace=True, errors="ignore")
            # Forcer l'id en string
            if AccountCols.ID in accounts_df.columns:
                accounts_df[AccountCols.ID] = accounts_df[AccountCols.ID].astype(str).str.replace(r"^u", "", regex=True)
                
            # Convert date
            if AccountCols.CREATED_AT in accounts_df.columns:
                accounts_df[AccountCols.CREATED_AT] = pd.to_datetime(accounts_df[AccountCols.CREATED_AT], errors="coerce", utc=True)
                
        # 3. Posts (Seulement tweet_0.json si présent, pour éviter l'OOM)
        posts_df = None
        tweet_files = sorted(p.glob("tweet_*.json"))
        if tweet_files:
            target_tweet = tweet_files[0]
            logger.info(f"[{self.name}] Chargement des tweets depuis {target_tweet.name}...")
            posts_df = pd.read_json(target_tweet, orient="records", nrows=nrows)
            
            p_renames = {
                "id": PostCols.ID,
                "author_id": PostCols.ACCOUNT_ID,
                "text": PostCols.TEXT,
                "created_at": PostCols.CREATED_AT
            }
            posts_df.rename(columns=p_renames, inplace=True, errors="ignore")
            
            if PostCols.ACCOUNT_ID in posts_df.columns:
                posts_df[PostCols.ACCOUNT_ID] = posts_df[PostCols.ACCOUNT_ID].astype(str).str.replace(r"^u", "", regex=True)
            if PostCols.CREATED_AT in posts_df.columns:
                posts_df[PostCols.CREATED_AT] = pd.to_datetime(posts_df[PostCols.CREATED_AT], errors="coerce", utc=True)

        # 4. Edges
        edges_df = None
        edge_file = p / "edge.csv"
        if edge_file.exists():
            edges_df = pd.read_csv(edge_file, nrows=nrows)
            # source,target,relation
            edges_df.rename(columns={"source": "source_account_id", "target": "target_account_id", "relation": "edge_type"}, inplace=True, errors="ignore")
            if "source_account_id" in edges_df.columns:
                edges_df["source_account_id"] = edges_df["source_account_id"].astype(str).str.replace(r"^u", "", regex=True)
                edges_df["target_account_id"] = edges_df["target_account_id"].astype(str).str.replace(r"^u", "", regex=True)

        # Si user_id n'est pas str dans labels, on force
        if label_df is not None and LabelCols.ACCOUNT_ID in label_df.columns:
            label_df[LabelCols.ACCOUNT_ID] = label_df[LabelCols.ACCOUNT_ID].astype(str).str.replace(r"^u", "", regex=True)

        return DataBundle(
            accounts_df=accounts_df,
            posts_df=posts_df,
            edges_df=edges_df,
            labels_df=label_df,
            source_path=str(p.resolve()),
            source_format="twibot-22",
            n_accounts=len(accounts_df) if accounts_df is not None else 0,
            n_posts=len(posts_df) if posts_df is not None else 0
        )

    @classmethod
    def can_handle(cls, path: Union[str, Path]) -> bool:
        """
        Détecte automatiquement si c'est un dossier TwiBot-22.
        """
        p = Path(path)
        if not p.is_dir():
            return False
        # Cherche les signatures TwiBot
        return (p / "user.json").exists() and (p / "label.csv").exists()
