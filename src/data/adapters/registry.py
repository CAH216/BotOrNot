# -*- coding: utf-8 -*-
"""
src/data/adapters/registry.py
-----------------------------
Registre central des adaptateurs de datasets.
Permet d'instancier dynamiquement l'adaptateur adéquat selon le paramètre `--adapter`.
"""

from typing import Union, Optional
from pathlib import Path

from src.data.adapters.base_adapter import BaseAdapter
from src.data.adapters.flatfile_adapter import FlatFileAdapter
from src.data.adapters.twibot_adapter import TwiBot22Adapter
from src.data.adapters.historical_adapter import HistoricalAdapter

ADAPTERS = {
    "flat-file": FlatFileAdapter,
    "twibot-22": TwiBot22Adapter,
    "historical": HistoricalAdapter
}

def get_adapter(adapter_name: str = "auto", path: Optional[Union[str, Path]] = None) -> BaseAdapter:
    """
    Retourne l'instance de l'adaptateur requis ou détecté.
    """
    if adapter_name != "auto" and adapter_name in ADAPTERS:
        return ADAPTERS[adapter_name]()
    
    # Auto detection
    if path is not None:
        for name, adapter_cls in ADAPTERS.items():
            if adapter_cls.can_handle(path):
                return adapter_cls()
                
    # Fallback par défaut (Legacy behavior)
    return FlatFileAdapter()
