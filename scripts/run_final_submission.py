#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_final_submission.py
========================
Script unique d'exécution Jour J.

Usage :
  python scripts/run_final_submission.py \\
      --en  dataset/dataset.posts&users.NEW_EN.json \\
      --fr  dataset/dataset.posts&users.NEW_FR.json \\
      --team MonTeam

Génère :
  submissions/MonTeam.detections.en.txt
  submissions/MonTeam.detections.fr.txt
  submissions/MonTeam.report.json

En cas d'erreur technique sur le champion, bascule automatiquement
sur le fallback correspondant et le signale clairement.
"""

import argparse
import json
import sys
import os
import time
import hashlib
import warnings
import traceback
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

from src.pipeline.monolithic_extractor import extract_monolithic_features
from src.features.forensic_humanness   import extract_forensic_humanness
from src.features.candidate_miner_court import CandidateMiner, PairwiseCourt, run_appeal_pipeline
from src.champion_config import (
    EN_MINER_CONFIG, EN_COURT_CONFIG,
    EN_FALLBACK_MINER_CONFIG, EN_FALLBACK_COURT_CONFIG,
    FR_MINER_CONFIG, FR_COURT_CONFIG,
    FR_FALLBACK_MINER_CONFIG, FR_FALLBACK_COURT_CONFIG,
    MONOLITH_CONFIG, EN_TRAIN_EVENTS, FR_TRAIN_EVENTS,
    get_lgbm_params, get_kfold_params, get_fr_synthetic_data,
)


# ─── Chargement ──────────────────────────────────────────────────────────────────

def load_json_dataset(json_path: str) -> tuple:
    """Charge un dataset JSON et retourne (users_df, posts_df, metadata)."""
    print(f"  📂 Chargement : {json_path}")
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    u = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
    p = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
    u["user_id"] = u["user_id"].astype(str)
    p["user_id"] = p["user_id"].astype(str)

    # Traiter z-score e-17 comme ≈0 (cf. note champion_config.py)
    for col in u.select_dtypes(include=[np.number]).columns:
        u[col] = u[col].fillna(0)
    for col in p.select_dtypes(include=[np.number]).columns:
        p[col] = p[col].fillna(0)

    meta = d.get("metadata", {})
    n_users = len(u)
    print(f"     → {n_users} comptes | {len(p)} posts | metadata={bool(meta)}")
    return u, p, meta


def load_historical_events(event_ids: list, dataset_dir: str = "dataset") -> list:
    """Charge tous les events d'entraînement historiques."""
    train_data = []
    for n in event_ids:
        jp = f"{dataset_dir}/dataset.posts&users.{n}.json"
        bp = f"{dataset_dir}/dataset.bots.{n}.txt"
        if not os.path.exists(jp):
            print(f"  ⚠  Event E{n} introuvable — ignoré.")
            continue
        with open(jp, encoding="utf-8") as f:
            d = json.load(f)
        u = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
        p = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
        u["user_id"] = u["user_id"].astype(str)
        p["user_id"] = p["user_id"].astype(str)
        with open(bp, encoding="utf-8") as f:
            bots = {s.strip() for s in f if s.strip()}
        u["is_bot"] = u["user_id"].isin(bots).astype(int)
        meta = d.get("metadata", {})
        feat   = extract_monolithic_features(u, p, meta, config=MONOLITH_CONFIG)
        feat_n = feat.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
        labels = u.set_index("user_id")["is_bot"]
        train_data.append({"feat": feat_n, "labels": labels, "n": n})
        print(f"     E{n} ✓ ({len(u)} comptes, {int(u['is_bot'].sum())} bots)")
    return train_data


# ─── Pipeline de prédiction ───────────────────────────────────────────────────────

def predict_bots(
    u_df, p_df, meta,
    train_data: list,
    miner_cfg: dict,
    court_cfg: dict,
    synth_df=None, synth_y=None,
    label: str = "?",
) -> tuple:
    """
    Prédit les bots pour un dataset de test.
    Retourne (bot_ids_list, n_accounts, n_bots_predicted, probs_dict).
    """
    lgbm_p = get_lgbm_params()
    kf_p   = get_kfold_params()

    # Features du dataset de test
    feat_te  = extract_monolithic_features(u_df, p_df, meta, config=MONOLITH_CONFIG)
    feat_te  = feat_te.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
    foren_te = extract_forensic_humanness(u_df, p_df).set_index("user_id")
    uids     = list(feat_te.index)

    # Matrice d'entraînement (tous events historiques)
    X_tr = pd.concat([t["feat"] for t in train_data])
    y_tr = np.concatenate([t["labels"].values for t in train_data])
    cols = [c for c in X_tr.columns if c in feat_te.columns]

    # Augmentation synthétique si fournie
    if synth_df is not None:
        sc = [c for c in cols if c in synth_df.columns]
        if sc:
            X_sa = pd.DataFrame(0.0, index=synth_df.index, columns=cols)
            for c in sc:
                X_sa[c] = synth_df[c].values
            X_tr = pd.concat([X_tr[cols], X_sa[cols]])
            y_tr = np.concatenate([y_tr, synth_y])
            print(f"     Synth ajouté : {len(synth_df)} comptes → train total={len(X_tr)}")

    # Ensemble LightGBM
    kf    = StratifiedKFold(**kf_p)
    probs = np.zeros(len(uids))
    for fold_i, (tri, _) in enumerate(kf.split(X_tr, y_tr)):
        m = LGBMClassifier(**lgbm_p)
        m.fit(X_tr.iloc[tri], y_tr[tri])
        probs += m.predict_proba(feat_te[cols])[:, 1] / kf.n_splits
    print(f"     LightGBM 5-fold ✓ | prob moy={probs.mean():.3f} | prob max={probs.max():.3f}")

    # Court d'appel
    y_dummy = pd.Series(np.zeros(len(uids)), index=uids)
    miner   = CandidateMiner(**miner_cfg)
    court   = PairwiseCourt(**court_cfg)
    preds, appeal_log = run_appeal_pipeline(
        uids, probs, feat_te[cols], foren_te, y_dummy, cols,
        miner=miner, court=court
    )

    bot_ids = [uid for uid, pred in zip(uids, preds) if pred == 1]
    n_actions = sum(1 for a in appeal_log.values() if a.get("action", "none") != "none")
    print(f"     Court d'appel ✓ | {n_actions} action(s) | {len(bot_ids)} bots prédits")

    probs_dict = dict(zip(uids, probs.tolist()))
    return bot_ids, len(uids), len(bot_ids), probs_dict


# ─── Écriture des sorties ─────────────────────────────────────────────────────────

def write_detections(bot_ids: list, output_path: str):
    """Écrit la liste des bot IDs, un par ligne, encodage UTF-8."""
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for uid in bot_ids:
            f.write(f"{uid}\n")
    print(f"  💾 Fichier écrit : {output_path} ({len(bot_ids)} IDs)")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


# ─── Main ─────────────────────────────────────────────────────────────────────────

def run_lang(
    lang: str, json_path: str, team: str,
    submissions_dir: Path,
    dataset_dir: str = "dataset",
    use_fallback: bool = False,
) -> dict:
    """Exécute le pipeline pour une langue. Retourne le rapport."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  🔵 Pipeline {lang} {'[FALLBACK]' if use_fallback else '[CHAMPION]'}")
    print(f"{'='*60}")

    # Config
    if lang == "EN":
        train_events    = EN_TRAIN_EVENTS
        miner_cfg       = EN_FALLBACK_MINER_CONFIG if use_fallback else EN_MINER_CONFIG
        court_cfg       = EN_FALLBACK_COURT_CONFIG if use_fallback else EN_COURT_CONFIG
        synth_df, synth_y = None, None
    else:
        train_events    = FR_TRAIN_EVENTS
        miner_cfg       = FR_FALLBACK_MINER_CONFIG if use_fallback else FR_MINER_CONFIG
        court_cfg       = FR_FALLBACK_COURT_CONFIG if use_fallback else FR_COURT_CONFIG
        if not use_fallback:
            print(f"\n  Génération synthétique FR v2...")
            synth_df, synth_y = get_fr_synthetic_data()
            print(f"     {len(synth_df)} comptes synthétiques FR générés ✓")
        else:
            synth_df, synth_y = None, None

    config_name = "fallback" if use_fallback else "champion"
    output_path = str(submissions_dir / f"{team}.detections.{lang.lower()}.txt")

    # Chargement dataset test
    print(f"\n  📥 Dataset test {lang} :")
    u_df, p_df, meta = load_json_dataset(json_path)

    # Chargement training historique
    print(f"\n  📚 Training historique {lang} (events {train_events}) :")
    train_data = load_historical_events(train_events, dataset_dir)
    if not train_data:
        raise RuntimeError(f"Aucun event d'entraînement trouvé pour {lang}!")

    # Prédiction
    print(f"\n  ⚙️  Prédiction {lang} ({config_name})...")
    bot_ids, n_accounts, n_bots, probs_dict = predict_bots(
        u_df, p_df, meta, train_data,
        miner_cfg, court_cfg, synth_df, synth_y, label=lang
    )

    # Écriture
    print(f"\n  ✍️  Écriture output...")
    write_detections(bot_ids, output_path)

    elapsed = round(time.time() - t0, 1)
    sha     = file_sha256(output_path)

    return {
        "lang":         lang,
        "config":       config_name,
        "json_path":    str(json_path),
        "output_path":  output_path,
        "n_accounts":   n_accounts,
        "n_bots":       n_bots,
        "pct_bots":     round(n_bots / n_accounts * 100, 1) if n_accounts else 0,
        "file_sha256":  sha,
        "elapsed_sec":  elapsed,
        "timestamp":    datetime.now().isoformat(),
        "miner_config": miner_cfg,
        "court_config": court_cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="BotOrNot — Exécution finale Jour J")
    parser.add_argument("--en",   required=True,  help="Chemin JSON dataset EN")
    parser.add_argument("--fr",   required=True,  help="Chemin JSON dataset FR")
    parser.add_argument("--team", required=True,  help="Nom de l'équipe (ex: MonTeam)")
    parser.add_argument("--fallback-en", action="store_true", help="Forcer fallback EN")
    parser.add_argument("--fallback-fr", action="store_true", help="Forcer fallback FR")
    parser.add_argument("--dataset-dir", default="dataset", help="Répertoire datasets historiques")
    parser.add_argument("--out-dir", default="submissions", help="Répertoire de sortie")
    args = parser.parse_args()

    global_start = time.time()
    submissions_dir = Path(args.out_dir)
    submissions_dir.mkdir(exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  🏆 BotOrNot — Run Final Soumission")
    print(f"  Équipe   : {args.team}")
    print(f"  Démarré  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    report = {"team": args.team, "langs": {}, "total_elapsed_sec": 0}

    # ── Pipeline EN ──────────────────────────────────────────────────────────────
    fallback_en = args.fallback_en
    try:
        result_en = run_lang(
            "EN", args.en, args.team, submissions_dir,
            dataset_dir=args.dataset_dir, use_fallback=fallback_en
        )
        report["langs"]["EN"] = result_en
    except Exception as e:
        if not fallback_en:
            print(f"\n  ⚠️  CHAMPION EN ÉCHOUÉ : {e}")
            print(f"  🔄 Basculement automatique sur FALLBACK EN...")
            traceback.print_exc()
            result_en = run_lang(
                "EN", args.en, args.team, submissions_dir,
                dataset_dir=args.dataset_dir, use_fallback=True
            )
            result_en["fallback_reason"] = str(e)
            report["langs"]["EN"] = result_en
        else:
            raise

    # ── Pipeline FR ──────────────────────────────────────────────────────────────
    fallback_fr = args.fallback_fr
    try:
        result_fr = run_lang(
            "FR", args.fr, args.team, submissions_dir,
            dataset_dir=args.dataset_dir, use_fallback=fallback_fr
        )
        report["langs"]["FR"] = result_fr
    except Exception as e:
        if not fallback_fr:
            print(f"\n  ⚠️  CHAMPION FR ÉCHOUÉ : {e}")
            print(f"  🔄 Basculement automatique sur FALLBACK FR...")
            traceback.print_exc()
            result_fr = run_lang(
                "FR", args.fr, args.team, submissions_dir,
                dataset_dir=args.dataset_dir, use_fallback=True
            )
            result_fr["fallback_reason"] = str(e)
            report["langs"]["FR"] = result_fr
        else:
            raise

    # ── Rapport final ─────────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - global_start, 1)
    report["total_elapsed_sec"] = total_elapsed

    report_path = submissions_dir / f"{args.team}.report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'#'*60}")
    print(f"  ✅ SOUMISSION TERMINÉE")
    print(f"{'#'*60}")
    for lang, r in report["langs"].items():
        fb = " [FALLBACK]" if r.get("config") == "fallback" else ""
        print(f"  {lang}{fb}: {r['n_bots']} bots / {r['n_accounts']} comptes ({r['pct_bots']}%)")
        print(f"       Fichier : {r['output_path']}")
        print(f"       SHA256  : {r['file_sha256']}")
        print(f"       Temps   : {r['elapsed_sec']}s")
    print(f"\n  Rapport JSON : {report_path}")
    print(f"  Temps total  : {total_elapsed}s ({total_elapsed/60:.1f} min)")
    print(f"\n  📧 Fichiers à attacher à l'email :")
    for lang, r in report["langs"].items():
        print(f"     → {Path(r['output_path']).name}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
