#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_residual_surgery.py
==============================
Mission : Court Leverage (Veto protecteur & Nomination étendue E30, Veto FR).

Compare :
1. Champion actuel
2. + EN Court Veto (E5 Rescue)
3. + EN Court-driven Nomination (E30 Rescue)
4. + Tout combiné EN
5. Champion FR actuel
6. + FR Court Veto (E6 FP Rescue)
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
from src.features.candidate_miner_court import (
    CandidateMiner, PairwiseCourt, run_appeal_pipeline
)

CFG = {"use_vas": True, "use_lrh": True, "use_lrh2": True}

TARGET_EN = ["7ad32a09-f784-49c6-b490-63c4ed63061d", # no_context_poll (FP)
             "b8e3dd3a-f0ae-49d4-8904-91791d4488d0", # rise_quotes (FP)
             "0cc2606e-eeb4-4db3-b6b8-983c52e0051b", # sapalocha98 (FP)
             "3cc9a093-6c8a-4db5-9017-f584e0306c5d", # MLarkinHockey (FP)
             "389c9d5a-5284-41d3-9f89-8bded8087fc7", # TravelWithEm (FN)
             "cdb1853d-2562-4299-add3-8c8eb2e9ba4d", # KKBello__ (FN)
             "75fa3d17-9154-4a25-9f5b-5fc18600a747"] # JordanAlways_ (FN)

TARGET_FR = ["1388ae1b-7aee-4c6e-9407-3532f641a0be"] # pete_prk (FP)

def load_event(n):
    jp = f"dataset/dataset.posts&users.{n}.json"
    bp = f"dataset/dataset.bots.{n}.txt"
    if not os.path.exists(jp): return None
    with open(jp, encoding="utf-8") as f: d = json.load(f)
    u = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
    p = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
    u["user_id"] = u["user_id"].astype(str)
    p["user_id"] = p["user_id"].astype(str)
    with open(bp, encoding="utf-8") as f: bots = {s.strip() for s in f if s.strip()}
    u["is_bot"] = u["user_id"].isin(bots).astype(int)
    return u, p, d.get("metadata", {}), bots

def section(t, w=70): print(f"\n{'='*w}\n  {t}\n{'='*w}")

def official(yt, yp):
    tp=int(((yp==1)&(yt==1)).sum()); fn=int(((yp==0)&(yt==1)).sum())
    fp=int(((yp==1)&(yt==0)).sum()); return tp,fn,fp,2*tp-2*fn-6*fp

def bs_p5(scores, n=200, seed=42):
    rng = np.random.default_rng(seed)
    return round(float(np.percentile(
        [np.mean(rng.choice(scores, len(scores), replace=True)) for _ in range(n)], 5)), 1)

def get_name(uid, u_df):
    r = u_df[u_df["user_id"]==uid]
    return str(r.iloc[0].get("username", "?")) if not r.empty else uid[:8]

def precompute(events):
    cache = {}
    for n, (u, p, m, bots) in events.items():
        feat = extract_monolithic_features(u, p, m, config=CFG)
        feat_num = feat.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
        forensic = extract_forensic_humanness(u, p).set_index("user_id")
        labels = u.set_index("user_id")["is_bot"]
        cache[n] = {"feat": feat_num, "forensic": forensic, "labels": labels, "u": u, "p": p, "bots": bots}
    return cache

def run_loeo(cache, miner_cfg=None, court_cfg=None):
    results = []
    use_court = miner_cfg is not None
    
    for test_n in cache:
        X_te = cache[test_n]["feat"]
        F_te = cache[test_n]["forensic"]
        y_s  = cache[test_n]["labels"]
        uids = list(X_te.index)
        y_te = y_s.loc[uids].values

        trains = [cache[k] for k in cache if k != test_n]
        X_tr = pd.concat([t["feat"] for t in trains])
        y_tr = np.concatenate([t["labels"].values for t in trains])
        cols = [c for c in X_tr.columns if c in X_te.columns]
        
        probs = np.zeros(len(uids))
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tri, _ in kf.split(X_tr[cols], y_tr):
            m = LGBMClassifier(random_state=42, verbose=-1, n_estimators=150)
            m.fit(X_tr[cols].iloc[tri], y_tr[tri])
            probs += m.predict_proba(X_te[cols])[:, 1] / kf.n_splits
            
        appeal_log = {}
        if use_court:
            miner = CandidateMiner(**miner_cfg)
            court = PairwiseCourt(**court_cfg)
            preds, appeal_log = run_appeal_pipeline(uids, probs, X_te[cols], F_te, pd.Series(y_te, index=uids), cols, miner=miner, court=court)
        else:
            preds = (probs >= 0.5).astype(int)
            
        tp, fn, fp, sc = official(y_te, preds)
        results.append({
            "event": test_n, "tp": tp, "fn": fn, "fp": fp, "score": sc,
            "y_true": y_te, "y_pred": preds, "uids": np.array(uids), "appeal": appeal_log
        })
    return results

def print_cfg(label, results, ref=None):
    sc = [r["score"] for r in results]
    print(f"\n  [{label}]  mean={np.mean(sc):.1f}  p5={bs_p5(sc)}  FP_max={max(r['fp'] for r in results)}")
    for r in sorted(results, key=lambda x: x["event"]):
        d = ""
        if ref:
            rr = next(x for x in ref if x["event"]==r["event"])
            diff = r["score"] - rr["score"]
            d = f"  ({'+' if diff>=0 else ''}{diff})"
        print(f"    E{r['event']}: {r['score']:4d}  TP={r['tp']} FN={r['fn']} FP={r['fp']}{d}")

def get_target_status(results_list, cache, targets):
    ret = []
    for test_n in cache:
        r = next(x for x in results_list if x["event"] == test_n)
        u_df = cache[test_n]["u"]
        uids = list(r["uids"])
        for uid in targets:
            if uid in uids:
                i = uids.index(uid)
                is_bot = r["y_true"][i] == 1
                pred = r["y_pred"][i]
                app = r["appeal"].get(uid, {})
                ret.append({
                    "name": get_name(uid, u_df),
                    "event": test_n,
                    "is_bot": is_bot,
                    "pred": pred,
                    "action": app.get("action", "none"),
                    "bot_votes": app.get("bot_votes", "?")
                })
    return ret

def main():
    en_raw = {n: load_event(n) for n in [1, 3, 5, 30] if load_event(n)}
    fr_raw = {n: load_event(n) for n in [2, 4, 6, 31] if load_event(n)}
    
    section("PRÉ-CALCUL")
    en_c = precompute(en_raw); fr_c = precompute(fr_raw)
    
    # configs EN
    C_EN_BASE = dict(use_expansion=False, use_veto=False, proba_low=0.01, proba_high=0.35, forensic_percentile=65, human_archetype_cap=0.30)
    C_EN_VETO = dict(use_expansion=False, use_veto=True,  proba_low=0.01, proba_high=0.35, forensic_percentile=65, human_archetype_cap=0.30)
    C_EN_EXP  = dict(use_expansion=True,  use_veto=False, proba_low=0.01, proba_high=0.35, forensic_percentile=65, human_archetype_cap=0.30)
    C_EN_ALL  = dict(use_expansion=True,  use_veto=True,  proba_low=0.01, proba_high=0.35, forensic_percentile=65, human_archetype_cap=0.30)
    COURT_EN  = dict(k=3, min_bot_votes=2)
    
    # configs FR
    C_FR_BASE = dict(use_expansion=False, use_veto=False, proba_low=0.01, proba_high=0.50, forensic_percentile=50, human_archetype_cap=0.30)
    C_FR_VETO = dict(use_expansion=False, use_veto=True,  proba_low=0.01, proba_high=0.50, forensic_percentile=50, human_archetype_cap=0.30)
    COURT_FR  = dict(k=3, min_bot_votes=3)
    
    section("ÉTAPE 1 — BENCHMARK COMBINÉ")
    
    r_en_1 = run_loeo(en_c, C_EN_BASE, COURT_EN)
    r_en_2 = run_loeo(en_c, C_EN_VETO, COURT_EN)
    r_en_3 = run_loeo(en_c, C_EN_EXP,  COURT_EN)
    r_en_4 = run_loeo(en_c, C_EN_ALL,  COURT_EN)
    
    r_fr_1 = run_loeo(fr_c, C_FR_BASE, COURT_FR)
    r_fr_2 = run_loeo(fr_c, C_FR_VETO, COURT_FR)
    
    print("\n=== BRANCHE EN ===")
    print_cfg("1. Champion EN Actuel (Rescue Only)", r_en_1)
    print_cfg("2. Champion + EN Court Veto", r_en_2, r_en_1)
    print_cfg("3. Champion + EN E30 Expansion", r_en_3, r_en_1)
    print_cfg("4. TOUT COMBINÉ EN (Veto + Exp)", r_en_4, r_en_1)
    
    print("\n=== BRANCHE FR ===")
    print_cfg("5. Champion FR Actuel", r_fr_1)
    print_cfg("6. Champion FR + FR Court Veto", r_fr_2, r_fr_1)
    
    section("ÉTAPE 2 — FOCUS SUR LES CIBLES RÉSIDUELLES")
    
    stat_en = get_target_status(r_en_4, en_c, TARGET_EN)
    print("  === Bilan EN-ALL sur les Cibles EN ===")
    for s in stat_en:
        tag = "TP✅" if (s["is_bot"] and s["pred"]==1) else ("TN✅" if (not s["is_bot"] and s["pred"]==0) else ("FN❌" if s["is_bot"] else "FP❌"))
        print(f"    @{s['name']:<18} (E{s['event']:<2}) : {tag:<4} | Court Votes: {s['bot_votes']} | Action: {s['action']}")
        
    stat_fr = get_target_status(r_fr_2, fr_c, TARGET_FR)
    print("\n  === Bilan FR-VETO sur la Cible FR ===")
    for s in stat_fr:
        tag = "TP✅" if (s["is_bot"] and s["pred"]==1) else ("TN✅" if (not s["is_bot"] and s["pred"]==0) else ("FN❌" if s["is_bot"] else "FP❌"))
        print(f"    @{s['name']:<18} (E{s['event']:<2}) : {tag:<4} | Court Votes: {s['bot_votes']} | Action: {s['action']}")
        
    section("VERDICT FINAL")
    m_en = np.mean([r["score"] for r in r_en_4])
    m_fr = np.mean([r["score"] for r in r_fr_2])
    fp_en = max(r["fp"] for r in r_en_4)
    fp_fr = max(r["fp"] for r in r_fr_2)
    
    print(f"  Score EN => {m_en:.1f}  (FP_max {fp_en})")
    print(f"  Score FR => {m_fr:.1f}  (FP_max {fp_fr})")
    
    if m_en > 100.0 and fp_en == 0 and m_fr > 45.5 and fp_fr == 0:
        print("  🟢 APPROUVÉ : Residual Surgery complète réussie !")
    else:
        print("  🤔 ANALYSE REQUISE : Vérifiez les impacts.")

if __name__ == "__main__":
    main()
