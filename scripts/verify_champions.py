#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_champions.py
====================
Vérification finale officielle des deux champions.
Lit la configuration depuis src/champion_config.py et rejoue les deux benchmarks
dans les mêmes conditions exactes que les benchmarks de référence.

Attendu :
  EN : mean=101.5  p5=94.9  FP_max=1
  FR : mean=47.5   p5=41.0  FP_max=1
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
    MONOLITH_CONFIG, OFFICIAL_SCORES,
    get_lgbm_params, get_kfold_params,
)
from scripts.benchmark_fr_dual import generate_fr_hard_negatives_v2


def load_event(n):
    jp = f"dataset/dataset.posts&users.{n}.json"
    bp = f"dataset/dataset.bots.{n}.txt"
    if not os.path.exists(jp): return None
    with open(jp, encoding="utf-8") as f: d = json.load(f)
    u = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
    p = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
    u["user_id"] = u["user_id"].astype(str); p["user_id"] = p["user_id"].astype(str)
    with open(bp, encoding="utf-8") as f: bots = {s.strip() for s in f if s.strip()}
    u["is_bot"] = u["user_id"].isin(bots).astype(int)
    return u, p, d.get("metadata", {}), bots

def precompute(events):
    cache = {}
    for n, (u, p, m, bots) in events.items():
        feat  = extract_monolithic_features(u, p, m, config=MONOLITH_CONFIG)
        feat_n = feat.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
        foren  = extract_forensic_humanness(u, p).set_index("user_id")
        labels = u.set_index("user_id")["is_bot"]
        cache[n] = {"feat": feat_n, "forensic": foren, "labels": labels, "u": u}
    return cache

def official(yt, yp):
    tp=int(((yp==1)&(yt==1)).sum()); fn=int(((yp==0)&(yt==1)).sum())
    fp=int(((yp==1)&(yt==0)).sum()); return tp,fn,fp,2*tp-2*fn-6*fp

def bs_p5(scores, n=200, seed=42):
    rng = np.random.default_rng(seed)
    return round(float(np.percentile(
        [np.mean(rng.choice(scores, len(scores), replace=True)) for _ in range(n)], 5)), 1)

def run_loeo(cache, miner_cfg, court_cfg, synth_df=None, synth_y=None):
    lgbm_p = get_lgbm_params()
    kf_p   = get_kfold_params()
    results = []

    for test_n in cache:
        X_te = cache[test_n]["feat"]
        F_te = cache[test_n]["forensic"]
        y_s  = cache[test_n]["labels"]
        uids = list(X_te.index)
        y_te = y_s.loc[uids].values

        trains = [cache[k] for k in cache if k != test_n]
        X_tr = pd.concat([t["feat"]   for t in trains])
        y_tr = np.concatenate([t["labels"].values for t in trains])
        cols = [c for c in X_tr.columns if c in X_te.columns]

        # Augmentation synthétique (FR uniquement)
        if synth_df is not None:
            synth_cols = [c for c in cols if c in synth_df.columns]
            if synth_cols:
                X_synth_a = pd.DataFrame(0.0, index=synth_df.index, columns=cols)
                for c in synth_cols:
                    X_synth_a[c] = synth_df[c].values
                X_tr_a = pd.concat([X_tr[cols], X_synth_a[cols]])
                y_tr_a = np.concatenate([y_tr, synth_y])
            else:
                X_tr_a, y_tr_a = X_tr[cols], y_tr
        else:
            X_tr_a, y_tr_a = X_tr[cols], y_tr

        kf = StratifiedKFold(**kf_p)
        probs = np.zeros(len(uids))
        for tri, _ in kf.split(X_tr_a, y_tr_a):
            m = LGBMClassifier(**lgbm_p)
            m.fit(X_tr_a.iloc[tri], y_tr_a[tri])
            probs += m.predict_proba(X_te[cols])[:, 1] / kf.n_splits

        miner = CandidateMiner(**miner_cfg)
        court = PairwiseCourt(**court_cfg)
        preds, _ = run_appeal_pipeline(
            uids, probs, X_te[cols], F_te,
            pd.Series(y_te, index=uids), cols,
            miner=miner, court=court
        )
        tp, fn, fp, sc = official(y_te, preds)
        results.append({"event": test_n, "tp": tp, "fn": fn, "fp": fp, "score": sc})
    return results

def check(label, results, expected_mean, expected_p5, expected_fp_max):
    sc  = [r["score"] for r in results]
    m   = round(np.mean(sc), 1)
    p5  = bs_p5(sc)
    fpm = max(r["fp"] for r in results)
    ok_m  = abs(m   - expected_mean)   <= 2.0
    ok_p5 = abs(p5  - expected_p5)     <= 3.0
    ok_fp = fpm <= expected_fp_max
    status = "✅ VALIDÉ" if (ok_m and ok_fp) else "❌ ÉCHOUÉ"
    print(f"\n  [{label}]  {status}")
    print(f"    mean={m:.1f} (attendu ≈{expected_mean})  {'✅' if ok_m else '❌'}")
    print(f"    p5  ={p5}   (attendu ≈{expected_p5})   {'✅' if ok_p5 else '❌'}")
    print(f"    FP_max={fpm} (max autorisé={expected_fp_max})  {'✅' if ok_fp else '❌'}")
    for r in sorted(results, key=lambda x: x["event"]):
        print(f"    E{r['event']:2d}: {r['score']:4d} | TP={r['tp']} FN={r['fn']} FP={r['fp']}")
    return ok_m and ok_fp

def section(t, w=70): print(f"\n{'='*w}\n  {t}\n{'='*w}")

def main():
    print("\n🏆 Vérification Finale Officielle — Champions BotOrNot")
    print(   "   Source de vérité : src/champion_config.py")

    en_raw = {n: load_event(n) for n in [1, 3, 5, 30] if load_event(n)}
    fr_raw = {n: load_event(n) for n in [2, 4, 6, 31] if load_event(n)}

    section("PRÉ-CALCUL")
    en_c = precompute(en_raw)
    fr_c = precompute(fr_raw)
    print("  Done.")

    section("GÉNÉRATION SYNTHÉTIQUE FR v2")
    synth_df, synth_y = generate_fr_hard_negatives_v2(
        n_per_bot   = 50,
        n_per_human = 60,
        seed        = 42,
    )
    print(f"  {len(synth_df)} comptes synthétiques FR v2 générés.")

    section("BENCHMARK EN")
    print("\n  Running Champion EN (Veto, proba_high=0.35, forensic_p=65)...")
    r_en = run_loeo(en_c, EN_MINER_CONFIG, EN_COURT_CONFIG)

    section("BENCHMARK FR")
    print("\n  Running Champion FR (+ Synth FR v2)...")
    r_fr = run_loeo(fr_c, FR_MINER_CONFIG, FR_COURT_CONFIG,
                    synth_df=synth_df, synth_y=synth_y)

    section("VÉRIFICATION OFFICIELLE")
    en_exp = OFFICIAL_SCORES["EN"]
    fr_exp = OFFICIAL_SCORES["FR"]

    ok_en = check("EN Champion", r_en,
                  en_exp["mean"], en_exp["p5"], en_exp["FP_max"])
    ok_fr = check("FR Champion + Synth v2", r_fr,
                  fr_exp["mean"], fr_exp["p5"], fr_exp["FP_max"])

    section("RÉSULTAT FINAL")
    if ok_en and ok_fr:
        print("\n  🟢 LES DEUX CHAMPIONS SONT VALIDÉS ET PRÊTS POUR LA COMPÉTITION.")
        print(f"  EN : mean≈{en_exp['mean']}  p5≈{en_exp['p5']}  FP_max={en_exp['FP_max']}")
        print(f"  FR : mean≈{fr_exp['mean']}  p5≈{fr_exp['p5']}  FP_max={fr_exp['FP_max']}")
    elif ok_en:
        print("\n  🟡 Champion EN validé. Champion FR à recalibrer.")
    elif ok_fr:
        print("\n  🟡 Champion FR validé. Champion EN à recalibrer.")
    else:
        print("\n  🔴 ATTENTION — Aucun champion validé. Vérifier les modifications récentes.")

if __name__ == "__main__":
    main()
