#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
score_competition_real.py
==========================
Simulation compétition bout-en-bout.

Protocole exact compétition :
  - EN : train sur E1 + E3 + E30 → prédit sur E5 (E5 = dataset inconnu)
  - FR : train sur E2 + E4 + E31 → prédit sur E6 (E6 = dataset inconnu)
  - Génération des fichiers de soumission via run_final_submission logique
  - Comparaison aux ground truth dataset.bots.5.txt et dataset.bots.6.txt
  - Score officiel : 2*TP - 2*FN - 6*FP

Aucun LOEO, aucune cross-validation, une seule simulation.
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.monolithic_extractor import extract_monolithic_features
from src.features.forensic_humanness   import extract_forensic_humanness
from src.features.candidate_miner_court import CandidateMiner, PairwiseCourt, run_appeal_pipeline
from src.champion_config import (
    EN_MINER_CONFIG, EN_COURT_CONFIG,
    FR_MINER_CONFIG, FR_COURT_CONFIG,
    MONOLITH_CONFIG, get_lgbm_params, get_kfold_params,
    get_fr_synthetic_data,
)

# ─── Configuration simulation ──────────────────────────────────────────────────
SIM_EN_TEST  = 5          # Dataset EN simulant la finale
SIM_FR_TEST  = 6          # Dataset FR simulant la finale
SIM_EN_TRAIN = [1, 3, 30] # Tous EN historiques SAUF le dataset test
SIM_FR_TRAIN = [2, 4, 31] # Tous FR historiques SAUF le dataset test
TEAM_NAME    = "RealCompSim"
OUT_DIR      = Path("submissions_real")
DATASET_DIR  = "dataset"


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def load_event(n: int) -> dict:
    jp = f"{DATASET_DIR}/dataset.posts&users.{n}.json"
    bp = f"{DATASET_DIR}/dataset.bots.{n}.txt"
    with open(jp, encoding="utf-8") as f: d = json.load(f)
    u  = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
    p  = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
    u["user_id"] = u["user_id"].astype(str)
    p["user_id"] = p["user_id"].astype(str)
    with open(bp, encoding="utf-8") as f:
        bots = {s.strip() for s in f if s.strip()}
    u["is_bot"] = u["user_id"].isin(bots).astype(int)
    return {"u": u, "p": p, "meta": d.get("metadata", {}), "bots": bots, "n": n}


def extract_features(ev: dict) -> dict:
    feat   = extract_monolithic_features(ev["u"], ev["p"], ev["meta"], config=MONOLITH_CONFIG)
    feat_n = feat.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
    foren  = extract_forensic_humanness(ev["u"], ev["p"]).set_index("user_id")
    labels = ev["u"].set_index("user_id")["is_bot"]
    return {"feat": feat_n, "forensic": foren, "labels": labels}


def predict(
    test_feat, test_foren,
    train_list,
    miner_cfg, court_cfg,
    synth_df=None, synth_y=None,
) -> tuple:
    lgbm_p = get_lgbm_params(); kf_p = get_kfold_params()
    uids   = list(test_feat.index)

    X_tr = pd.concat([t["feat"] for t in train_list])
    y_tr = np.concatenate([t["labels"].values for t in train_list])
    cols = [c for c in X_tr.columns if c in test_feat.columns]

    if synth_df is not None:
        sc = [c for c in cols if c in synth_df.columns]
        if sc:
            X_sa = pd.DataFrame(0.0, index=synth_df.index, columns=cols)
            for c in sc: X_sa[c] = synth_df[c].values
            X_tr = pd.concat([X_tr[cols], X_sa[cols]])
            y_tr = np.concatenate([y_tr, synth_y])

    kf    = StratifiedKFold(**kf_p)
    probs = np.zeros(len(uids))
    for tri, _ in kf.split(X_tr[cols], y_tr):
        m = LGBMClassifier(**lgbm_p)
        m.fit(X_tr[cols].iloc[tri], y_tr[tri])
        probs += m.predict_proba(test_feat[cols])[:, 1] / kf.n_splits

    y_dummy = pd.Series(np.zeros(len(uids)), index=uids)
    miner   = CandidateMiner(**miner_cfg)
    court   = PairwiseCourt(**court_cfg)
    preds, appeal_log = run_appeal_pipeline(
        uids, probs, test_feat[cols], test_foren, y_dummy, cols,
        miner=miner, court=court
    )

    prob_map = dict(zip(uids, probs.tolist()))
    return np.array(preds), prob_map, appeal_log, uids


def official_score(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp, fn, fp, 2 * tp - 2 * fn - 6 * fp


def write_detections(bot_ids, path):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for uid in bot_ids: f.write(f"{uid}\n")


def get_username(u_df, uid):
    row = u_df[u_df["user_id"] == uid]
    if not row.empty and "username" in row.columns:
        return str(row["username"].values[0])
    return uid[:10]


# ─── Rapport ──────────────────────────────────────────────────────────────────

def section(title, w=70):
    print(f"\n{'═'*w}\n  {title}\n{'═'*w}")


def print_report(
    lang, test_ev,
    y_true, y_pred, preds_array,
    prob_map, appeal_log, uids,
    elapsed,
):
    u_df = test_ev["u"]
    uids_arr = np.array(uids)
    y_true_arr = np.array(y_true)

    tp, fn, fp, sc = official_score(y_true_arr, preds_array)
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    n_bots_real = int(y_true_arr.sum())
    n_bots_pred = int(preds_array.sum())
    n_total     = len(uids)

    print(f"\n  {'─'*60}")
    print(f"  {lang} — E{test_ev['n']}")
    print(f"  {'─'*60}")
    print(f"  Comptes total   : {n_total}")
    print(f"  Bots réels      : {n_bots_real}")
    print(f"  Bots prédits    : {n_bots_pred}")
    print(f"")
    print(f"  TP              : {tp}")
    print(f"  FN              : {fn}  (bots manqués)")
    print(f"  FP              : {fp}  (humains accusés)")
    print(f"")
    print(f"  Précision       : {precision:.1f}%")
    print(f"  Recall          : {recall:.1f}%")
    print(f"  {'─'*40}")
    print(f"  🎯 SCORE {lang}     : {sc:+d}  (2×{tp} - 2×{fn} - 6×{fp})")
    print(f"  Temps pipeline  : {elapsed:.1f}s")

    # FN : bots manqués
    fn_uids = [uid for uid, yt, yp in zip(uids, y_true_arr, preds_array)
               if yt == 1 and yp == 0]
    # FP : humains accusés
    fp_uids = [uid for uid, yt, yp in zip(uids, y_true_arr, preds_array)
               if yt == 0 and yp == 1]

    if fn_uids:
        print(f"\n  🔴 FN {lang} — Bots manqués ({len(fn_uids)}) :")
        for uid in fn_uids:
            uname = get_username(u_df, uid)
            prob  = round(prob_map.get(uid, -1), 3)
            app   = appeal_log.get(uid, {})
            action = app.get("action", "none"); votes = app.get("bot_votes", "?")
            print(f"    @{uname:<22} prob={prob:.3f}  court={action}({votes})")

    if fp_uids:
        print(f"\n  🟡 FP {lang} — Humains accusés ({len(fp_uids)}) :")
        for uid in fp_uids:
            uname = get_username(u_df, uid)
            prob  = round(prob_map.get(uid, -1), 3)
            app   = appeal_log.get(uid, {})
            action = app.get("action", "none"); votes = app.get("bot_votes", "?")
            print(f"    @{uname:<22} prob={prob:.3f}  court={action}({votes})")

    return sc, tp, fn, fp, fn_uids, fp_uids


def main():
    t_global = time.time()

    print("\n" + "█"*70)
    print("  🏆  SIMULATION COMPÉTITION RÉELLE — Score Officiel")
    print("  Protocole : entraîner sur historique, prédire sur dataset test unique")
    print("█"*70)
    print(f"\n  EN test  : E{SIM_EN_TEST}  | Entraîné sur : {SIM_EN_TRAIN}")
    print(f"  FR test  : E{SIM_FR_TEST}  | Entraîné sur : {SIM_FR_TRAIN}")

    # ── Chargement datasets ──────────────────────────────────────────────────
    section("Chargement")
    print("  Test events...")
    en_test = load_event(SIM_EN_TEST)
    fr_test = load_event(SIM_FR_TEST)
    print(f"  E{SIM_EN_TEST} EN : {len(en_test['u'])} comptes, {len(en_test['bots'])} bots réels")
    print(f"  E{SIM_FR_TEST} FR : {len(fr_test['u'])} comptes, {len(fr_test['bots'])} bots réels")

    print("\n  Training events...")
    en_trains_raw = [load_event(n) for n in SIM_EN_TRAIN]
    fr_trains_raw = [load_event(n) for n in SIM_FR_TRAIN]
    for ev in en_trains_raw:
        print(f"  E{ev['n']} EN-train : {len(ev['u'])} comptes, {len(ev['bots'])} bots")
    for ev in fr_trains_raw:
        print(f"  E{ev['n']} FR-train : {len(ev['u'])} comptes, {len(ev['bots'])} bots")

    # ── Feature extraction ───────────────────────────────────────────────────
    section("Feature Extraction")
    print("  Test EN..."); en_test_f  = extract_features(en_test)
    print("  Test FR..."); fr_test_f  = extract_features(fr_test)
    print("  Train EN...")
    en_trains = [extract_features(ev) for ev in en_trains_raw]
    print("  Train FR...")
    fr_trains = [extract_features(ev) for ev in fr_trains_raw]
    print("  Synth FR v2...")
    synth_df, synth_y = get_fr_synthetic_data()
    print(f"  {len(synth_df)} comptes synthétiques FR ✓")

    # ── Prédiction EN ────────────────────────────────────────────────────────
    section("Pipeline EN — Champion")
    t_en = time.time()
    preds_en, probs_en, appeal_en, uids_en = predict(
        en_test_f["feat"], en_test_f["forensic"],
        en_trains, EN_MINER_CONFIG, EN_COURT_CONFIG,
    )
    elapsed_en = time.time() - t_en
    y_true_en  = en_test_f["labels"].loc[uids_en].values
    bot_ids_en = [uid for uid, p in zip(uids_en, preds_en) if p == 1]

    out_en = OUT_DIR / f"{TEAM_NAME}.detections.en.txt"
    write_detections(bot_ids_en, out_en)
    print(f"  ✓ {len(bot_ids_en)} bots prédits  → {out_en}")

    # ── Prédiction FR ────────────────────────────────────────────────────────
    section("Pipeline FR — Champion + Synth v2")
    t_fr = time.time()
    preds_fr, probs_fr, appeal_fr, uids_fr = predict(
        fr_test_f["feat"], fr_test_f["forensic"],
        fr_trains, FR_MINER_CONFIG, FR_COURT_CONFIG,
        synth_df=synth_df, synth_y=synth_y,
    )
    elapsed_fr = time.time() - t_fr
    y_true_fr  = fr_test_f["labels"].loc[uids_fr].values
    bot_ids_fr = [uid for uid, p in zip(uids_fr, preds_fr) if p == 1]

    out_fr = OUT_DIR / f"{TEAM_NAME}.detections.fr.txt"
    write_detections(bot_ids_fr, out_fr)
    print(f"  ✓ {len(bot_ids_fr)} bots prédits  → {out_fr}")

    # ── RAPPORT FINAL ─────────────────────────────────────────────────────────
    section("RAPPORT SCORE COMPÉTITION RÉEL")

    sc_en, tp_en, fn_en, fp_en, fn_uids_en, fp_uids_en = print_report(
        "EN", en_test, en_test_f["labels"].loc[uids_en].values,
        None, preds_en, probs_en, appeal_en, uids_en, elapsed_en
    )
    sc_fr, tp_fr, fn_fr, fp_fr, fn_uids_fr, fp_uids_fr = print_report(
        "FR", fr_test, fr_test_f["labels"].loc[uids_fr].values,
        None, preds_fr, probs_fr, appeal_fr, uids_fr, elapsed_fr
    )

    total = sc_en + sc_fr
    total_tp = tp_en + tp_fr
    total_fn = fn_en + fn_fr
    total_fp = fp_en + fp_fr
    total_elapsed = time.time() - t_global

    print(f"\n{'═'*70}")
    print(f"  🏆 SCORE TOTAL COMPÉTITION : {total}")
    print(f"     EN : {sc_en:+d}  |  FR : {sc_fr:+d}")
    print(f"     TP={total_tp}  FN={total_fn}  FP={total_fp}")
    print(f"     Recall global   : {total_tp/(total_tp+total_fn)*100:.1f}%")
    print(f"     Précision glob. : {total_tp/(total_tp+total_fp)*100:.1f}%" if (total_tp+total_fp)>0 else "     Précision glob. : 100.0%")
    print(f"     Temps total     : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'═'*70}")

    # Comptes les plus intéressants à analyser
    interesting = []
    for uid in fn_uids_en:
        uname = get_username(en_test["u"], uid)
        interesting.append({"account": f"@{uname}", "type": "FN EN", "prob": round(probs_en.get(uid, 0), 3)})
    for uid in fp_uids_en:
        uname = get_username(en_test["u"], uid)
        interesting.append({"account": f"@{uname}", "type": "FP EN", "prob": round(probs_en.get(uid, 0), 3)})
    for uid in fn_uids_fr:
        uname = get_username(fr_test["u"], uid)
        interesting.append({"account": f"@{uname}", "type": "FN FR", "prob": round(probs_fr.get(uid, 0), 3)})
    for uid in fp_uids_fr:
        uname = get_username(fr_test["u"], uid)
        interesting.append({"account": f"@{uname}", "type": "FP FR", "prob": round(probs_fr.get(uid, 0), 3)})

    if interesting:
        print(f"\n  📌 Comptes à analyser ({len(interesting)}) :")
        print(f"  {'Compte':<26} {'Type':<10} {'Prob'}")
        print(f"  {'─'*46}")
        for r in sorted(interesting, key=lambda x: abs(x["prob"] - 0.5)):
            print(f"  {r['account']:<26} {r['type']:<10} {r['prob']:.3f}")

    print(f"\n  📂 Fichiers générés :")
    print(f"     {out_en}  ({len(bot_ids_en)} IDs)")
    print(f"     {out_fr}  ({len(bot_ids_fr)} IDs)")
    print()


if __name__ == "__main__":
    main()
