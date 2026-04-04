#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
competition_simulation.py
==========================
Simulation fidèle de la compétition.

Protocole exact :
  - Pour chaque event test, entraîner sur TOUS les autres events.
  - Prédire sur l'event test avec le champion (miner + court).
  - Calculer le score officiel : 2*TP - 2*FN - 6*FP par event.
  - Additionner tous les scores pour le total final.

Events simulés :
  EN : E1, E3, E5, E30
  FR : E2, E4, E6, E31

Résultat affiché comme un leaderboard de compétition.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.monolithic_extractor import extract_monolithic_features
from src.features.forensic_humanness   import extract_forensic_humanness
from src.features.candidate_miner_court import CandidateMiner, PairwiseCourt, run_appeal_pipeline
from src.champion_config import (
    EN_MINER_CONFIG, EN_COURT_CONFIG,
    FR_MINER_CONFIG, FR_COURT_CONFIG,
    MONOLITH_CONFIG, get_lgbm_params, get_kfold_params,
)
from scripts.benchmark_fr_dual import generate_fr_hard_negatives_v2


def load_event(n):
    jp = f"dataset/dataset.posts&users.{n}.json"
    bp = f"dataset/dataset.bots.{n}.txt"
    if not os.path.exists(jp):
        return None
    with open(jp, encoding="utf-8") as f:
        d = json.load(f)
    u = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
    p = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
    u["user_id"] = u["user_id"].astype(str)
    p["user_id"] = p["user_id"].astype(str)
    with open(bp, encoding="utf-8") as f:
        bots = {s.strip() for s in f if s.strip()}
    u["is_bot"] = u["user_id"].isin(bots).astype(int)
    lang = "FR" if n in [2, 4, 6, 31] else "EN"
    return {"u": u, "p": p, "meta": d.get("metadata", {}), "bots": bots, "lang": lang, "n": n}


def precompute(event_data, cfg):
    ev = event_data
    feat   = extract_monolithic_features(ev["u"], ev["p"], ev["meta"], config=cfg)
    feat_n = feat.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
    foren  = extract_forensic_humanness(ev["u"], ev["p"]).set_index("user_id")
    labels = ev["u"].set_index("user_id")["is_bot"]
    return {"feat": feat_n, "forensic": foren, "labels": labels, "u": ev["u"], "lang": ev["lang"]}


def official_score(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp, fn, fp, 2 * tp - 2 * fn - 6 * fp


def run_competition(all_cache, miner_cfg_by_lang, court_cfg_by_lang,
                    fr_synth_df=None, fr_synth_y=None):
    """
    Pour chaque event, entraîne sur tous les autres et prédit.
    Retourne la liste des résultats par event.
    """
    lgbm_p = get_lgbm_params()
    kf_p   = get_kfold_params()
    results = []

    for test_n, test_data in all_cache.items():
        lang     = test_data["lang"]
        X_te     = test_data["feat"]
        F_te     = test_data["forensic"]
        y_s      = test_data["labels"]
        u_df     = test_data["u"]
        uids     = list(X_te.index)
        y_te     = y_s.loc[uids].values

        # Entraîner sur TOUS les autres events (EN + FR mélangés = plus de données)
        trains   = [v for k, v in all_cache.items() if k != test_n]
        X_tr     = pd.concat([t["feat"] for t in trains])
        y_tr     = np.concatenate([t["labels"].values for t in trains])
        cols     = [c for c in X_tr.columns if c in X_te.columns]

        # Augmentation FR synthétique si event FR
        if lang == "FR" and fr_synth_df is not None:
            sc = [c for c in cols if c in fr_synth_df.columns]
            if sc:
                X_sa = pd.DataFrame(0.0, index=fr_synth_df.index, columns=cols)
                for c in sc: X_sa[c] = fr_synth_df[c].values
                X_tr = pd.concat([X_tr[cols], X_sa[cols]])
                y_tr = np.concatenate([y_tr, fr_synth_y])

        # Ensemble 5-fold LightGBM
        kf    = StratifiedKFold(**kf_p)
        probs = np.zeros(len(uids))
        for tri, _ in kf.split(X_tr, y_tr):
            m = LGBMClassifier(**lgbm_p)
            m.fit(X_tr.iloc[tri], y_tr[tri])
            probs += m.predict_proba(X_te[cols])[:, 1] / kf.n_splits

        # Pairwise Court  (config selon langue)
        miner = CandidateMiner(**miner_cfg_by_lang[lang])
        court = PairwiseCourt(**court_cfg_by_lang[lang])
        preds, appeal_log = run_appeal_pipeline(
            uids, probs, X_te[cols], F_te,
            pd.Series(y_te, index=uids), cols,
            miner=miner, court=court
        )

        tp, fn, fp, sc = official_score(y_te, preds)

        # Détail des comptes changés par le court
        changed = []
        for uid, app in appeal_log.items():
            action = app.get("action", "none")
            if action != "none":
                uname = u_df[u_df["user_id"] == uid]["username"].values
                uname = uname[0] if len(uname) else uid[:8]
                label = int(y_s.loc[uid])
                changed.append({
                    "name": uname, "label": label,
                    "prob": round(float(probs[uids.index(uid)]), 3),
                    "action": action,
                    "bot_votes": app.get("bot_votes", "?"),
                })

        results.append({
            "event": test_n, "lang": lang,
            "n_accounts": len(uids),
            "n_bots": int(y_te.sum()),
            "n_humans": int((y_te == 0).sum()),
            "tp": tp, "fn": fn, "fp": fp, "score": sc,
            "changed": changed,
        })

    return results


def print_leaderboard(results):
    SEP = "=" * 72

    en_results = [r for r in results if r["lang"] == "EN"]
    fr_results = [r for r in results if r["lang"] == "FR"]

    en_total = sum(r["score"] for r in en_results)
    fr_total = sum(r["score"] for r in fr_results)
    grand_total = en_total + fr_total

    print(f"\n{SEP}")
    print(f"  🏆  SIMULATION COMPÉTITION — RÉSULTATS OFFICIELS")
    print(f"{SEP}")

    print(f"\n  {'Event':<6} {'Lang':<5} {'Comptes':<9} {'Bots':<6} {'TP':<5} {'FN':<5} {'FP':<5} {'SCORE':>7}")
    print(f"  {'-'*66}")

    for r in sorted(results, key=lambda x: (x["lang"], x["event"])):
        bot_acc = r["tp"] / r["n_bots"] * 100 if r["n_bots"] > 0 else 0
        fp_tag  = " ⚠" if r["fp"] > 0 else "  "
        print(f"  E{r['event']:<5} {r['lang']:<5} {r['n_accounts']:<9} "
              f"{r['n_bots']:<6} {r['tp']:<5} {r['fn']:<5} {r['fp']:<5} "
              f"{r['score']:>6}{fp_tag}")

    print(f"  {'-'*66}")
    print(f"  {'EN TOTAL':<34}{'':>26} {en_total:>6}")
    print(f"  {'FR TOTAL':<34}{'':>26} {fr_total:>6}")
    print(f"  {SEP[:66]}")
    print(f"  {'🏆 SCORE TOTAL COMPÉTITION':<34}{'':>26} {grand_total:>6}")
    print(f"{SEP}")

    # Métriques de précision
    print(f"\n  📊 Précision de détection :")
    all_tp = sum(r["tp"] for r in results)
    all_fn = sum(r["fn"] for r in results)
    all_fp = sum(r["fp"] for r in results)
    all_bots = sum(r["n_bots"] for r in results)
    recall    = all_tp / (all_tp + all_fn) * 100 if (all_tp + all_fn) > 0 else 0
    precision = all_tp / (all_tp + all_fp) * 100 if (all_tp + all_fp) > 0 else 0
    print(f"    Recall  (bots trouvés)  : {recall:.1f}%  ({all_tp}/{all_bots})")
    print(f"    Précision (parmi prédits bots) : {precision:.1f}%")
    print(f"    Faux positifs totaux    : {all_fp}")
    print(f"    Faux négatifs totaux    : {all_fn}")

    # Détail court d'appel
    all_changed = [c for r in results for c in r["changed"]]
    if all_changed:
        print(f"\n  ⚖️  Court d'Appel — {len(all_changed)} compte(s) modifié(s) :")
        for r in sorted(results, key=lambda x: x["event"]):
            for c in r["changed"]:
                truth = "🤖 BOT  " if c["label"] == 1 else "👤 HUMAN"
                action_tag = "rescuté ✅" if (c["action"] == "rescue" and c["label"] == 0) else \
                             "nommé ✅"   if (c["action"] == "nominate" and c["label"] == 1) else \
                             "vetoed ⚠"  if (c["action"] == "veto" and c["label"] == 0) else \
                             f"{c['action']}"
                print(f"    E{r['event']} | {truth} @{c['name']:<20} "
                      f"prob={c['prob']:.3f} votes={c['bot_votes']} [{action_tag}]")
    print()


def main():
    print("\n🎯 Chargement des datasets compétition (E1→E31)...")

    all_events_raw = {}
    for n in [1, 2, 3, 4, 5, 6, 30, 31]:
        ev = load_event(n)
        if ev:
            all_events_raw[n] = ev
            print(f"    E{n:2d} ({ev['lang']}) : {len(ev['u'])} comptes, "
                  f"{int(ev['u']['is_bot'].sum())} bots")

    print(f"\n🧬 Génération synthétique FR v2 (augmentation training)...")
    synth_df, synth_y = generate_fr_hard_negatives_v2(50, 60, 42)
    print(f"   {len(synth_df)} comptes synthétiques FR générés.")

    print(f"\n⚙️  Feature extraction (config champion)...")
    all_cache = {}
    for n, ev in all_events_raw.items():
        print(f"   Precomputing E{n}...", end=" ", flush=True)
        all_cache[n] = precompute(ev, MONOLITH_CONFIG)
        print("✓")

    miner_by_lang = {"EN": EN_MINER_CONFIG, "FR": FR_MINER_CONFIG}
    court_by_lang = {"EN": EN_COURT_CONFIG, "FR": FR_COURT_CONFIG}

    print(f"\n🏃 Simulation compétition (LOEO — entraîne sur tous sauf l'event testé)...")
    results = run_competition(
        all_cache,
        miner_cfg_by_lang = miner_by_lang,
        court_cfg_by_lang = court_by_lang,
        fr_synth_df = synth_df,
        fr_synth_y  = synth_y,
    )

    print_leaderboard(results)


if __name__ == "__main__":
    main()
