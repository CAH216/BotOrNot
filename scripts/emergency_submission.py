#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
emergency_submission.py
========================
PLAN DE SECOURS — Dernier recours si run_final_submission.py échoue.

Utilise uniquement LightGBM + seuil 0.5, sans court d'appel, sans synth.
Entraîne sur tous les events historiques disponibles.
Moins précis mais 0% de risque de crash.

Usage :
  python scripts/emergency_submission.py \\
      --en competition_day/dataset.posts&users.EN.json \\
      --fr competition_day/dataset.posts&users.FR.json \\
      --team NomEquipe
"""

import argparse, json, sys, os, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline.monolithic_extractor import extract_monolithic_features
from src.champion_config import MONOLITH_CONFIG, EN_TRAIN_EVENTS, FR_TRAIN_EVENTS


def load_json(path):
    with open(path, encoding="utf-8") as f: d = json.load(f)
    u = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
    p = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
    u["user_id"] = u["user_id"].astype(str); p["user_id"] = p["user_id"].astype(str)
    return u, p, d.get("metadata", {})


def emergency_predict(json_path, train_events, dataset_dir="dataset"):
    u, p, meta = load_json(json_path)
    feat_te = extract_monolithic_features(u, p, meta, config=MONOLITH_CONFIG)
    feat_te = feat_te.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
    uids = list(feat_te.index)

    trains = []
    for n in train_events:
        jp = f"{dataset_dir}/dataset.posts&users.{n}.json"
        bp = f"{dataset_dir}/dataset.bots.{n}.txt"
        if not os.path.exists(jp): continue
        with open(jp, encoding="utf-8") as f: d = json.load(f)
        ut = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
        pt = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
        ut["user_id"] = ut["user_id"].astype(str); pt["user_id"] = pt["user_id"].astype(str)
        with open(bp, encoding="utf-8") as f: bots = {s.strip() for s in f if s.strip()}
        ut["is_bot"] = ut["user_id"].isin(bots).astype(int)
        fe = extract_monolithic_features(ut, pt, d.get("metadata", {}), config=MONOLITH_CONFIG)
        fe = fe.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
        labels = ut.set_index("user_id")["is_bot"]
        trains.append({"feat": fe, "labels": labels})

    X_tr = pd.concat([t["feat"] for t in trains])
    y_tr = np.concatenate([t["labels"].values for t in trains])
    cols = [c for c in X_tr.columns if c in feat_te.columns]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros(len(uids))
    for tri, _ in kf.split(X_tr[cols], y_tr):
        m = LGBMClassifier(random_state=42, verbose=-1, n_estimators=150)
        m.fit(X_tr[cols].iloc[tri], y_tr[tri])
        probs += m.predict_proba(feat_te[cols])[:, 1] / kf.n_splits

    bot_ids = [uid for uid, prob in zip(uids, probs) if prob >= 0.5]
    return bot_ids, len(uids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", required=True); parser.add_argument("--fr", required=True)
    parser.add_argument("--team", required=True); parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    t0 = time.time()

    print("\n🚨 EMERGENCY SUBMISSION — Plan de secours activé")
    print("   (Sans court d'appel — seuil fixe 0.5)")

    for lang, json_path, events in [
        ("EN", args.en, EN_TRAIN_EVENTS),
        ("FR", args.fr, FR_TRAIN_EVENTS),
    ]:
        print(f"\n  Prédiction {lang}...")
        bot_ids, n_total = emergency_predict(json_path, events, args.dataset_dir)
        out_path = out / f"{args.team}.detections.{lang.lower()}.txt"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for uid in bot_ids: f.write(f"{uid}\n")
        print(f"  {lang}: {len(bot_ids)}/{n_total} bots → {out_path}")

    print(f"\n  ✅ Emergency terminé en {time.time()-t0:.1f}s")
    print(f"  📧 Attacher : {args.team}.detections.en.txt + {args.team}.detections.fr.txt")

if __name__ == "__main__":
    main()
