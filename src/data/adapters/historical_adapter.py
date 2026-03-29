# -*- coding: utf-8 -*-
"""
src/data/adapters/historical_adapter.py
---------------------------------------
Adaptateur spécifique pour le dataset pré-compétition (Event 30 ou 31).
Lit `dataset.posts&users.<id>.json` et fusionne avec `dataset.bots.<id>.txt`
pour produire le DataBundle avec labels.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from src.data.adapters.base_adapter import BaseAdapter
from src.data.schema import DataBundle, AccountCols, PostCols, LabelCols

logger = logging.getLogger(__name__)


class HistoricalAdapter(BaseAdapter):
    
    name = "historical"
    description = "Adaptateur natif pour les archives globales BotOrNot (Events 30, 31)."

    def load(
        self,
        base_path: Union[str, Path],
        nrows: Optional[int] = None,
        **kwargs
    ) -> DataBundle:
        """
        Déballe un fichier .posts&users.<num>.json et son jumeau .txt.
        """
        p = Path(base_path)
        if not p.is_file():
            raise FileNotFoundError(f"L'adaptateur historique attend un fichier JSON, reçu : {p}")
            
        logger.info(f"[{self.name}] Extraction depuis l'archive historique : {p.name}")
        
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "users" not in data or "posts" not in data:
            raise ValueError(f"'{p.name}' n'est pas un JSON historique valide (clés 'users' et 'posts' introuvables).")

        # 1. Parsing Accounts
        accounts_raw = data["users"][:nrows] if nrows else data["users"]
        accounts_df = pd.DataFrame(accounts_raw)
        if not accounts_df.empty:
            acc_renames = {
                "id": AccountCols.ID,
                "tweet_count": AccountCols.TOTAL_POSTS,
                "username": AccountCols.SCREEN_NAME,
                "description": AccountCols.BIO,
                "location": AccountCols.LOCATION
            }
            accounts_df.rename(columns=acc_renames, inplace=True, errors="ignore")
            # Force user_id as string
            if AccountCols.ID in accounts_df.columns:
                accounts_df[AccountCols.ID] = accounts_df[AccountCols.ID].astype(str)
        else:
            accounts_df = None

        # 2. Parsing Posts
        posts_raw = data["posts"][:nrows] if nrows else data["posts"]
        posts_df = pd.DataFrame(posts_raw)
        if not posts_df.empty:
            p_renames = {
                "id": PostCols.ID,
                "author_id": PostCols.ACCOUNT_ID,
                "text": PostCols.TEXT,
                "created_at": PostCols.CREATED_AT,
                "lang": PostCols.LANG,
                "source": PostCols.SOURCE
            }
            posts_df.rename(columns=p_renames, inplace=True, errors="ignore")
            if PostCols.ACCOUNT_ID in posts_df.columns:
                posts_df[PostCols.ACCOUNT_ID] = posts_df[PostCols.ACCOUNT_ID].astype(str)
            if PostCols.CREATED_AT in posts_df.columns:
                # Convert to datetime string format properly handled by loaders downstream if needed
                posts_df[PostCols.CREATED_AT] = pd.to_datetime(posts_df[PostCols.CREATED_AT], utc=True, errors="coerce")
        else:
            posts_df = None

        # 3. Deduction du fichier texte pour les labels
        label_df = None
        txt_name = p.name.replace(".posts&users.", ".bots.").replace(".json", ".txt")
        label_file = p.parent / txt_name
        
        if label_file.exists():
            logger.info(f"[{self.name}] Découverte du fichier des bots : {label_file.name}")
            with open(label_file, "r", encoding="utf-8") as f:
                bot_ids = {line.strip() for line in f if line.strip()}
                
            if accounts_df is not None and AccountCols.ID in accounts_df.columns:
                all_ids = accounts_df[AccountCols.ID].unique()
                labels_dict = {
                    uid: (1.0 if uid in bot_ids else 0.0)
                    for uid in all_ids
                }
                label_df = pd.DataFrame(list(labels_dict.items()), columns=[LabelCols.ACCOUNT_ID, LabelCols.LABEL])
        else:
            logger.warning(f"[{self.name}] Fichier TXT de tags introuvable : {txt_name}. Dataset non-labellisé.")

        return DataBundle(
            accounts_df=accounts_df,
            posts_df=posts_df,
            labels_df=label_df,
            source_path=str(p.resolve()),
            source_format="historical",
            n_accounts=len(accounts_df) if accounts_df is not None else 0,
            n_posts=len(posts_df) if posts_df is not None else 0
        )

    @classmethod
    def can_handle(cls, path: Union[str, Path]) -> bool:
        """
        Détecte automatiquement si c'est un fichier JSON d'Event historique (Ex: dataset.posts&users.30.json).
        """
        p = Path(path)
        return p.is_file() and ".posts&users." in p.name and p.suffix.lower() == ".json"
