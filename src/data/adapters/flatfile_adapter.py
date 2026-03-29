# -*- coding: utf-8 -*-
"""
src/data/adapters/flatfile_adapter.py
-------------------------------------
Adaptateur "Legacy" pour les fichiers plats actuels (CSV/JSON orientés colonnes).
Ré-encapsule la logique originelle de `src.data.loaders` sous le contrat BaseAdapter.
"""

from pathlib import Path
from typing import Optional, Union
import logging

from src.data.adapters.base_adapter import BaseAdapter
from src.data.schema import DataBundle
from src.data.loaders import load_file

logger = logging.getLogger(__name__)

class FlatFileAdapter(BaseAdapter):
    
    name = "flat-file"
    description = "Adaptateur standard pour CSV/JSON mono-fichier ou mixte."

    def load(
        self,
        base_path: Union[str, Path],
        nrows: Optional[int] = None,
        **kwargs
    ) -> DataBundle:
        """
        Délègue l'extraction au mécanisme original `load_file` qui gère
        déjà la séparation auto/accounts/posts.
        """
        logger.info(f"[{self.name}] Traitement via FlatFileAdapter: {base_path}")
        # On utilise le code legacy déjà très verouillé et fiable.
        return load_file(base_path, nrows=nrows)

    @classmethod
    def can_handle(cls, path: Union[str, Path]) -> bool:
        """
        S'active si c'est un fichier reconnu (.csv, .json, .jsonl).
        """
        p = Path(path)
        return p.is_file() and p.suffix.lower() in [".csv", ".json", ".jsonl"]
