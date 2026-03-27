# -*- coding: utf-8 -*-
"""
src/utils/timing.py
--------------------
Mesure du temps d'exécution et profilage des étapes du pipeline.

Utilitaires :
    Timer               → context manager ou décorateur pour mesurer une étape
    PipelineProfiler    → profiler cumulatif pour tout le run
    format_duration     → formatage lisible (2.3s / 1m4s / 3h12m)

Usage :
    from src.utils.timing import Timer, PipelineProfiler

    # Context manager
    with Timer("feature_extraction") as t:
        feat_df = extract_tabular_features(accounts_df, posts_df)
    print(f"Features : {t.elapsed_s:.2f}s")

    # Profiler cumulatif
    profiler = PipelineProfiler()
    with profiler.step("load_data"):
        bundle = load_bundle(path)
    with profiler.step("build_features"):
        X = assembler.assemble(bundle)
    print(profiler.report())
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes en chaîne lisible.

    Examples:
        0.45  → "450ms"
        2.3   → "2.3s"
        75    → "1m15s"
        3720  → "1h2m"
    """
    if seconds < 1.0:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m = rem // 60
        return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Timer — context manager simple
# ---------------------------------------------------------------------------

class Timer:
    """
    Chronomètre pour une étape du pipeline.

    Usage :
        with Timer("étape") as t:
            do_work()
        print(t.elapsed_s)
    """

    def __init__(
        self,
        name:    str  = "step",
        log:     bool = True,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.name      = name
        self.log       = log
        self._logger   = logger_ or logger
        self.elapsed_s = 0.0
        self._start    = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        if self.log:
            self._logger.debug("[⏱] Début : %s", self.name)
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_s = round(time.perf_counter() - self._start, 3)
        if self.log:
            self._logger.info("[⏱] %s : %s", self.name, format_duration(self.elapsed_s))

    def elapsed_ms(self) -> float:
        """Retourne le temps écoulé en millisecondes."""
        return round(self.elapsed_s * 1000, 1)


def timed(name: Optional[str] = None, log: bool = True) -> Callable:
    """
    Décorateur qui mesure le temps d'exécution d'une fonction.

    Usage :
        @timed("feature_extraction")
        def extract_features(df): ...
    """
    def decorator(func: Callable) -> Callable:
        step_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(step_name, log=log):
                return func(*args, **kwargs)

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# PipelineProfiler — profiler cumulatif
# ---------------------------------------------------------------------------

class StepRecord:
    """Enregistrement d'une étape de profiling."""
    def __init__(self, name: str, elapsed_s: float) -> None:
        self.name      = name
        self.elapsed_s = elapsed_s


class PipelineProfiler:
    """
    Profiler cumulatif pour mesurer toutes les étapes du pipeline.

    Usage :
        profiler = PipelineProfiler()
        with profiler.step("load"):
            ...
        with profiler.step("features"):
            ...
        print(profiler.report())
    """

    def __init__(self, name: str = "pipeline") -> None:
        self.name     = name
        self._records: List[StepRecord] = []
        self._start   = time.perf_counter()

    @contextmanager
    def step(self, step_name: str) -> Iterator[Timer]:
        """Mesure une étape et l'ajoute au profil."""
        t = Timer(step_name, log=True)
        with t:
            yield t
        self._records.append(StepRecord(step_name, t.elapsed_s))

    def total_s(self) -> float:
        """Temps total depuis la création du profiler."""
        return round(time.perf_counter() - self._start, 3)

    def step_times(self) -> Dict[str, float]:
        """Retourne un dict {étape → durée} trié par ordre d'exécution."""
        return {r.name: r.elapsed_s for r in self._records}

    def report(self) -> str:
        """Rapport textuel du profiling."""
        total = self.total_s()
        lines = [
            f"\n{'='*50}",
            f"  Profiling : {self.name}",
            f"  Total     : {format_duration(total)}",
            f"{'='*50}",
            f"  {'Étape':<28} {'Durée':>8} {'%':>6}",
            f"  {'-'*44}",
        ]
        for r in self._records:
            pct  = 100 * r.elapsed_s / max(total, 1e-9)
            bar  = "█" * int(pct / 5)
            lines.append(
                f"  {r.name:<28} {format_duration(r.elapsed_s):>8} {pct:>5.1f}%  {bar}"
            )
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Sérialise le profil en dict (pour RunLogger)."""
        return {
            "pipeline":   self.name,
            "total_s":    self.total_s(),
            "steps":      self.step_times(),
        }
