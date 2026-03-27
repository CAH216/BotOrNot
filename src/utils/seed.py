# -*- coding: utf-8 -*-
"""
src/utils/seed.py
------------------
Contrôle global du seed de reproductibilité.

Problème :
    Un modèle entraîné deux fois doit produire exactement les mêmes
    résultats si le seed est identique. Sans contrôle global, les librairies
    (numpy, random, sklearn, torch...) ont chacune leur propre état aléatoire.

Fonctions :
    set_global_seed(seed)  → fixe tous les rngs connus
    get_seed()             → retourne le seed actuel
    SeedContext            → context manager (reset automatique à la sortie)

Usage :
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    # ... tout le pipeline ...
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_CURRENT_SEED: Optional[int] = None


def set_global_seed(seed: int = 42) -> int:
    """
    Fixe le seed de toutes les sources d'aléatoire connues.

    Librairies ciblées :
        - Python random
        - NumPy
        - PyTorch (si installé)
        - TensorFlow (si installé)
        - CUDA (si installé)
        - Variable d'env PYTHONHASHSEED

    Args:
        seed : entier (0–2^32-1)

    Returns:
        Le seed appliqué
    """
    global _CURRENT_SEED
    _CURRENT_SEED = seed

    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch (optionnel)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Forcer le déterminisme (peut ralentir légèrement)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass

    # TensorFlow (optionnel)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    logger.debug("Seed global fixé à %d", seed)
    return seed


def get_seed() -> Optional[int]:
    """Retourne le seed actuellement fixé (None si jamais appelé)."""
    return _CURRENT_SEED


@contextmanager
def seed_context(seed: int) -> Iterator[None]:
    """
    Context manager : fixe le seed à l'entrée et restaure l'ancien à la sortie.

    Usage :
        with seed_context(123):
            model.fit(X, y)   # reproductible indépendamment du seed global
    """
    prev_seed = _CURRENT_SEED
    set_global_seed(seed)
    try:
        yield
    finally:
        if prev_seed is not None:
            set_global_seed(prev_seed)


def make_seeds(n: int, base_seed: int = 42) -> list[int]:
    """
    Génère N seeds reproductibles à partir d'un seed de base.
    Utile pour la cross-validation (un seed différent par fold).

    Args:
        n         : nombre de seeds à générer
        base_seed : seed de base

    Returns:
        Liste de N seeds entiers
    """
    rng = random.Random(base_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]
