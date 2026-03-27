# -*- coding: utf-8 -*-
"""
src/utils/logging.py
---------------------
Logging structuré pour BotOrNot.

Fonctionnalités :
    - Configuration du logging en une ligne (setup_logging)
    - Logger structuré par module avec niveaux cohérents
    - Sauvegarde automatique des configs et résultats d'un run dans un JSON
    - Résumé de run lisible en fin d'exécution

Usage :
    from src.utils.logging import setup_logging, RunLogger

    # Au démarrage du pipeline
    setup_logging(level="INFO", log_file="artifacts/logs/run.log")

    # Traçabilité d'un run
    run = RunLogger(run_id="run_001", config={"model": "lightgbm"})
    run.log_metrics({"auroc": 0.91, "f1": 0.87})
    run.save("artifacts/logs/run_001.json")
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Setup du logging global
# ---------------------------------------------------------------------------

def setup_logging(
    level:    str = "INFO",
    log_file: Optional[str] = None,
    fmt:      Optional[str] = None,
) -> None:
    """
    Configure le logging global du pipeline.

    Args:
        level    : niveau de log (DEBUG / INFO / WARNING / ERROR)
        log_file : chemin du fichier log (None = console seulement)
        fmt      : format personnalisé
    """
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"

    handlers: list = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level    = getattr(logging, level.upper(), logging.INFO),
        format   = fmt,
        datefmt  = "%Y-%m-%d %H:%M:%S",
        handlers = handlers,
        force    = True,
    )

    # Silencer les loggers tiers verbeux
    for noisy in ("urllib3", "filelock", "huggingface_hub", "httpx", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("BotOrNot").info("Logging initialisé — niveau=%s", level)


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé avec le namespace BotOrNot."""
    return logging.getLogger(f"BotOrNot.{name}")


# ---------------------------------------------------------------------------
# RunLogger — traçabilité complète d'un run
# ---------------------------------------------------------------------------

class RunLogger:
    """
    Journal structuré d'un run d'entraînement ou d'inférence.

    Chaque run reçoit un ID unique, une config, des métriques,
    et peut être sauvegardé en JSON pour comparaison.

    Usage :
        run = RunLogger.start(config={"model": "lightgbm", ...})
        run.log_metrics({"auroc": 0.91, "f1": 0.87, "threshold": 0.42})
        run.log_artifact("model", "artifacts/models/lgb_v1.joblib")
        run.finish()
        run.save("artifacts/logs/run_001.json")
        print(run.summary())
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id    = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.config    = config or {}
        self.metrics:   Dict[str, Any] = {}
        self.artifacts: Dict[str, str] = {}
        self.events:    list = []
        self.started_at  = datetime.utcnow().isoformat() + "Z"
        self.finished_at: Optional[str] = None
        self.elapsed_s:   Optional[float] = None
        self._start_ts   = datetime.utcnow().timestamp()
        self._logger     = get_logger("RunLogger")
        self._logger.info("Run démarré : %s", self.run_id)

    @classmethod
    def start(
        cls,
        config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> "RunLogger":
        """Factory method — démarre un nouveau run."""
        return cls(run_id=run_id, config=config)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Enregistre des métriques."""
        self.metrics.update(metrics)
        for k, v in metrics.items():
            self._logger.info("  %s = %s", k, v)

    def log_artifact(self, name: str, path: str) -> None:
        """Enregistre le chemin d'un artefact produit."""
        self.artifacts[name] = path
        self._logger.info("Artefact '%s' : %s", name, path)

    def log_event(self, event: str, data: Optional[dict] = None) -> None:
        """Enregistre un événement daté."""
        entry = {
            "ts":    datetime.utcnow().isoformat() + "Z",
            "event": event,
        }
        if data:
            entry.update(data)
        self.events.append(entry)
        self._logger.debug("Event : %s", event)

    def finish(self) -> float:
        """Marque la fin du run et retourne le temps écoulé."""
        self.finished_at = datetime.utcnow().isoformat() + "Z"
        self.elapsed_s   = round(datetime.utcnow().timestamp() - self._start_ts, 2)
        self._logger.info("Run terminé : %s (%.1fs)", self.run_id, self.elapsed_s)
        return self.elapsed_s

    def to_dict(self) -> dict:
        """Sérialise le run en dict."""
        return {
            "run_id":      self.run_id,
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s":   self.elapsed_s,
            "config":      self.config,
            "metrics":     self.metrics,
            "artifacts":   self.artifacts,
            "events":      self.events,
        }

    def save(self, path: str | Path) -> Path:
        """Sauvegarde le run en JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        self._logger.info("Run sauvegardé : %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunLogger":
        """Charge un run depuis un JSON."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        instance = cls(run_id=d.get("run_id"), config=d.get("config", {}))
        instance.metrics      = d.get("metrics", {})
        instance.artifacts    = d.get("artifacts", {})
        instance.events       = d.get("events", [])
        instance.started_at   = d.get("started_at")
        instance.finished_at  = d.get("finished_at")
        instance.elapsed_s    = d.get("elapsed_s")
        return instance

    def summary(self) -> str:
        """Résumé lisible du run."""
        lines = [
            f"\n{'='*55}",
            f"  Run  : {self.run_id}",
            f"  Start: {self.started_at}",
            f"  End  : {self.finished_at or '(en cours)'}",
            f"  Dur  : {self.elapsed_s or '?'}s",
            f"{'='*55}",
            "  Métriques :",
        ]
        for k, v in self.metrics.items():
            lines.append(f"    {k:<24} = {v}")
        if self.artifacts:
            lines.append("  Artefacts :")
            for k, v in self.artifacts.items():
                lines.append(f"    {k:<24} → {v}")
        lines.append("=" * 55)
        return "\n".join(lines)
