# -*- coding: utf-8 -*-
"""
src/data/adapters/base_adapter.py
---------------------------------
Interface de base pour les adaptateurs d'ingestion.
Chaque adaptateur doit pouvoir transformer un dossier ou un fichier source
en un objet DataBundle canonique.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
from pathlib import Path

from src.data.schema import DataBundle


class BaseAdapter(ABC):
    """
    Interface commune a tous les adaptateurs de datasets.
    """
    
    name = "base_adapter"
    description = "Adaptateur de base abstrait."

    @abstractmethod
    def load(
        self,
        base_path: Union[str, Path],
        nrows: Optional[int] = None,
        **kwargs
    ) -> DataBundle:
        """
        Lit les donnees depuis base_path et retourne un DataBundle normé.
        
        Args:
            base_path: Chemin principal (fichier ou dossier).
            nrows: Limite du nombre de lignes/elements a charger.
            kwargs: Arguments specifiques de l'adaptateur.
            
        Returns:
            DataBundle pret pour l'assemblage de features.
        """
        pass
    
    @classmethod
    def can_handle(cls, path: Union[str, Path]) -> bool:
        """
        Evalue si cet adaptateur sait gerer la cible.
        Par defaut abstrait, l'adaptateur doit inspecter le path.
        """
        return False
