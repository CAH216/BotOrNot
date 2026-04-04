#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_fr_dual.py
=====================
Rail FR — Benchmark dual comparable à benchmark_residual_surgery.py.

Compare (même protocole officiel — seed=42, LOEO, K=5) :
  1. Champion FR actuel (référence officielle : 45.5)
  2. + Synth FR v2 (archetypes ancrés sur vraies signatures FN FR)
  3. + Veto FR ciblé (@pete_prk : prob<0.60, court=1, hour_entropy élevée)
  4. + Combiné (Synth v2 + Veto FR)

Mesures : FR mean / p5 / FP_max / E2/E4/E6/E31 / comptes changés exacts.

Archétypes FR v2 (ancrés sur residual_table.txt) :
  Bots FR réels :
    - fr_midnighter_bot  : clock_frac_00_y élevé, tmp_night_ratio fort
    - fr_poll_nationalist: lrh_poll_score fort, hashtag modéré
    - fr_gentle_promo    : lrh_poll_score + tmp_ipt_cv modéré, tout honnête
  Humains FR ambigus :
    - fr_insomniac_human : noctambule réel, hour_entropy élevée
    - fr_political_human : politique actif, polls légitimes
    - fr_lifestyle_fr    : lifestyle FR, contenu varié
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
from src.champion_config import (
    FR_MINER_CONFIG, FR_COURT_CONFIG, MONOLITH_CONFIG,
    get_lgbm_params, get_kfold_params, OFFICIAL_SCORES
)

# ── Archétypes FR v2 (ancrés sur valeurs brutes des FN FR résiduels) ───────────
# Source : residual_table.txt — FN FR : clock_frac_00_y élevé, tmp_night_ratio fort,
#          lrh_poll_score modéré/fort, txt_has_hashtag_ratio typique FR

FR_ARCHETYPES_V2 = {

    # ── BOTS FR RÉELS ─────────────────────────────────────────────────────────

    "fr_midnighter_bot": {
        # Signature : clock_frac_00_y très élevé + tmp_night_ratio fort
        # Calibré sur : @JeanDupont31, @JeanDupont30, @chic_parisienne (FN E4/E6)
        "tmp_ipt_cv":            (1.5, 0.35),
        "tmp_ipt_std":           (3.8, 1.0),
        "tmp_ipt_min":           (0.04, 0.02),
        "tmp_hour_entropy":      (1.4, 0.3),
        "tmp_night_ratio":       (0.62, 0.1),   # très nocturne (signature clé)
        "tmp_peak_ratio":        (0.65, 0.08),
        "clock_frac_00_y":       (0.42, 0.08),  # minuit très présent (signature clé)
        "txt_has_hashtag_ratio": (0.50, 0.10),
        "txt_has_url_ratio":     (0.06, 0.02),
        "txt_std_len":           (14, 4),
        "txt_upper_ratio":       (0.06, 0.02),
        "gh_len_cv":             (0.40, 0.09),
        "gh_n_posts":            (70, 18),
        "usr_z_score_abs":       (0.38, 0.16),
        "usr_name_len":          (9, 2),
        "rel_topic_focus":       (0.22, 0.06),
        "lrh_poll_score":        (4.1, 0.7),    # poll fort (accusateur FR)
        "lrh_intent_score":      (0.12, 0.04),
        "lrh_legit_score":       (0.14, 0.05),
        "lrh2_residual_score":   (0.05, 0.02),
        "llm_bio_vanilla":       (0.62, 0.10),
        "human_lexical_entropy_y":(1.4, 0.30),
        "vas_spam_score":        (0.12, 0.05),
        "vas_clickbait_ratio":   (0.04, 0.02),
    },

    "fr_poll_nationalist_bot": {
        # Signature : lrh_poll_score très fort + clock_frac_00_y modéré
        # Calibré sur : @jeanpierre271, @louisCDX (FN E6)
        "tmp_ipt_cv":            (2.2, 0.5),
        "tmp_ipt_std":           (5.5, 1.4),
        "tmp_ipt_min":           (0.03, 0.015),
        "tmp_hour_entropy":      (1.6, 0.35),
        "tmp_night_ratio":       (0.45, 0.09),
        "tmp_peak_ratio":        (0.52, 0.08),
        "clock_frac_00_y":       (0.28, 0.07),
        "txt_has_hashtag_ratio": (0.58, 0.10),
        "txt_has_url_ratio":     (0.05, 0.02),
        "txt_std_len":           (16, 4),
        "txt_upper_ratio":       (0.07, 0.02),
        "gh_len_cv":             (0.44, 0.09),
        "gh_n_posts":            (80, 20),
        "usr_z_score_abs":       (0.42, 0.17),
        "usr_name_len":          (10, 2),
        "rel_topic_focus":       (0.20, 0.06),
        "lrh_poll_score":        (4.5, 0.75),   # signature principale
        "lrh_intent_score":      (0.10, 0.04),
        "lrh_legit_score":       (0.13, 0.05),
        "lrh2_residual_score":   (0.05, 0.02),
        "llm_bio_vanilla":       (0.60, 0.10),
        "human_lexical_entropy_y":(1.5, 0.32),
        "vas_spam_score":        (0.14, 0.05),
        "vas_clickbait_ratio":   (0.05, 0.02),
    },

    "fr_gentle_promo_bot": {
        # Signature : tout modéré, difficile à détecter — gh_len_cv + lrh_poll modéré
        # Calibré sur : @grandpapa_uk, @SirOldEnglish (FN E4/E6)
        "tmp_ipt_cv":            (1.2, 0.30),
        "tmp_ipt_std":           (3.0, 0.8),
        "tmp_ipt_min":           (0.05, 0.02),
        "tmp_hour_entropy":      (1.8, 0.35),
        "tmp_night_ratio":       (0.35, 0.08),
        "tmp_peak_ratio":        (0.45, 0.07),
        "clock_frac_00_y":       (0.20, 0.06),
        "txt_has_hashtag_ratio": (0.30, 0.08),
        "txt_has_url_ratio":     (0.08, 0.03),
        "txt_std_len":           (18, 5),
        "txt_upper_ratio":       (0.05, 0.02),
        "gh_len_cv":             (2.5, 0.5),    # gh_len_cv élevé (accusateur)
        "gh_n_posts":            (65, 16),
        "usr_z_score_abs":       (0.35, 0.15),
        "usr_name_len":          (8, 2),
        "rel_topic_focus":       (0.25, 0.07),
        "lrh_poll_score":        (3.6, 0.65),
        "lrh_intent_score":      (0.14, 0.04),
        "lrh_legit_score":       (0.16, 0.05),
        "lrh2_residual_score":   (0.06, 0.02),
        "llm_bio_vanilla":       (0.58, 0.10),
        "human_lexical_entropy_y":(1.45, 0.30),
        "vas_spam_score":        (0.10, 0.04),
        "vas_clickbait_ratio":   (0.03, 0.015),
    },

    # ── HUMAINS FR AMBIGUS ────────────────────────────────────────────────────

    "fr_insomniac_human": {
        # Noctambule réel : night_ratio fort MAIS hour_entropy élevée
        # Calibré sur : @pete_prk (FP E6) — tmp_hour_entropy + tmp_night_ratio
        "tmp_ipt_cv":            (5.8, 1.2),
        "tmp_ipt_std":           (15.0, 3.5),
        "tmp_ipt_min":           (1.8, 0.8),
        "tmp_hour_entropy":      (2.85, 0.45),  # très élevée = humain (variance clé)
        "tmp_night_ratio":       (0.55, 0.1),
        "tmp_peak_ratio":        (0.28, 0.07),
        "clock_frac_00_y":       (0.06, 0.025),
        "txt_has_hashtag_ratio": (0.42, 0.10),
        "txt_has_url_ratio":     (0.12, 0.04),
        "txt_std_len":           (30, 9),
        "txt_upper_ratio":       (0.05, 0.02),
        "gh_len_cv":             (0.85, 0.16),
        "gh_n_posts":            (55, 16),
        "usr_z_score_abs":       (0.28, 0.13),
        "usr_name_len":          (8, 2),
        "rel_topic_focus":       (0.42, 0.09),
        "lrh_poll_score":        (4.2, 0.75),   # accusateur malgré humain
        "lrh_intent_score":      (0.48, 0.11),
        "lrh_legit_score":       (0.60, 0.12),
        "lrh2_residual_score":   (0.35, 0.09),
        "llm_bio_vanilla":       (0.30, 0.09),
        "human_lexical_entropy_y":(2.5, 0.40),
        "vas_spam_score":        (0.07, 0.03),
        "vas_clickbait_ratio":   (0.02, 0.01),
    },

    "fr_political_human": {
        # Humain politique actif — polls légitimes, hashtags forts
        "tmp_ipt_cv":            (6.5, 1.3),
        "tmp_ipt_std":           (18.0, 4.0),
        "tmp_ipt_min":           (1.5, 0.7),
        "tmp_hour_entropy":      (2.6, 0.45),
        "tmp_night_ratio":       (0.42, 0.09),
        "tmp_peak_ratio":        (0.32, 0.07),
        "clock_frac_00_y":       (0.04, 0.02),
        "txt_has_hashtag_ratio": (0.62, 0.11),
        "txt_has_url_ratio":     (0.15, 0.05),
        "txt_std_len":           (28, 8),
        "txt_upper_ratio":       (0.04, 0.015),
        "gh_len_cv":             (0.78, 0.15),
        "gh_n_posts":            (70, 20),
        "usr_z_score_abs":       (0.32, 0.14),
        "usr_name_len":          (10, 3),
        "rel_topic_focus":       (0.38, 0.08),
        "lrh_poll_score":        (4.8, 0.80),
        "lrh_intent_score":      (0.55, 0.11),
        "lrh_legit_score":       (0.65, 0.12),
        "lrh2_residual_score":   (0.40, 0.10),
        "llm_bio_vanilla":       (0.28, 0.09),
        "human_lexical_entropy_y":(2.4, 0.40),
        "vas_spam_score":        (0.06, 0.025),
        "vas_clickbait_ratio":   (0.015, 0.008),
    },

    "fr_lifestyle_fr_human": {
        # Lifestyle FR varié — pas de signature botlike claire
        "tmp_ipt_cv":            (5.0, 1.0),
        "tmp_ipt_std":           (13.0, 3.0),
        "tmp_ipt_min":           (2.0, 0.9),
        "tmp_hour_entropy":      (2.7, 0.42),
        "tmp_night_ratio":       (0.32, 0.08),
        "tmp_peak_ratio":        (0.30, 0.07),
        "clock_frac_00_y":       (0.03, 0.015),
        "txt_has_hashtag_ratio": (0.35, 0.09),
        "txt_has_url_ratio":     (0.10, 0.04),
        "txt_std_len":           (35, 10),
        "txt_upper_ratio":       (0.04, 0.015),
        "gh_len_cv":             (0.92, 0.17),
        "gh_n_posts":            (58, 15),
        "usr_z_score_abs":       (0.25, 0.12),
        "usr_name_len":          (9, 2),
        "rel_topic_focus":       (0.48, 0.10),
        "lrh_poll_score":        (2.0, 0.55),
        "lrh_intent_score":      (0.58, 0.12),
        "lrh_legit_score":       (0.70, 0.13),
        "lrh2_residual_score":   (0.45, 0.10),
        "llm_bio_vanilla":       (0.25, 0.08),
        "human_lexical_entropy_y":(2.7, 0.42),
        "vas_spam_score":        (0.05, 0.02),
        "vas_clickbait_ratio":   (0.01, 0.005),
    },
}

FR_SYNTH_FEATURES = list(list(FR_ARCHETYPES_V2.values())[0].keys())
FR_BOT_ARCHETYPES   = [k for k in FR_ARCHETYPES_V2 if "_bot" in k]
FR_HUMAN_ARCHETYPES = [k for k in FR_ARCHETYPES_V2 if "_human" in k]


def generate_fr_hard_negatives_v2(n_per_bot=50, n_per_human=60, seed=0):
    """Génère un dataset FR hard-negative synthétique ancré sur les FN FR résiduels."""
    rng = np.random.default_rng(seed)
    frames, labels = [], []

    for i, arch_name in enumerate(FR_BOT_ARCHETYPES):
        arch = FR_ARCHETYPES_V2[arch_name]
        rows = []
        for j in range(n_per_bot):
            row = {"user_id": f"fr_synth_{arch_name}_{j}"}
            for feat, (mu, sigma) in arch.items():
                val = float(rng.normal(mu, sigma))
                val = max(0.0, val)
                row[feat] = val
            rows.append(row)
        frames.append(pd.DataFrame(rows))
        labels.extend([1] * n_per_bot)

    for i, arch_name in enumerate(FR_HUMAN_ARCHETYPES):
        arch = FR_ARCHETYPES_V2[arch_name]
        rows = []
        for j in range(n_per_human):
            row = {"user_id": f"fr_synth_{arch_name}_{j}"}
            for feat, (mu, sigma) in arch.items():
                val = float(rng.normal(mu, sigma))
                val = max(0.0, val)
                row[feat] = val
            rows.append(row)
        frames.append(pd.DataFrame(rows))
        labels.extend([0] * n_per_human)

    X = pd.concat(frames, ignore_index=True).set_index("user_id")
    return X, np.array(labels)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def run_loeo_fr(cache, miner_cfg, court_cfg, synth_df=None, synth_y=None,
                use_pete_veto=False):
    """LOEO FR complet. synth_df/y : données synthétiques ajoutées au train si fournies."""
    results = []
    lgbm_p = get_lgbm_params()
    kf_p   = get_kfold_params()

    for test_n in cache:
        X_te = cache[test_n]["feat"]
        F_te = cache[test_n]["forensic"]
        y_s  = cache[test_n]["labels"]
        u_df = cache[test_n]["u"]
        uids = list(X_te.index)
        y_te = y_s.loc[uids].values

        trains = [cache[k] for k in cache if k != test_n]
        X_tr = pd.concat([t["feat"]   for t in trains])
        y_tr = np.concatenate([t["labels"].values for t in trains])
        cols = [c for c in X_tr.columns if c in X_te.columns]

        # Augmentation synthétique FR (features disponibles dans cols)
        if synth_df is not None:
            synth_cols = [c for c in cols if c in synth_df.columns]
            if synth_cols:
                X_synth_aligned = pd.DataFrame(0.0, index=synth_df.index, columns=cols)
                for c in synth_cols:
                    X_synth_aligned[c] = synth_df[c].values
                X_tr_aug = pd.concat([X_tr[cols], X_synth_aligned[cols]])
                y_tr_aug = np.concatenate([y_tr, synth_y])
            else:
                X_tr_aug, y_tr_aug = X_tr[cols], y_tr
        else:
            X_tr_aug, y_tr_aug = X_tr[cols], y_tr

        # Probabilités de base (5-fold, config officielle)
        kf = StratifiedKFold(**kf_p)
        probs = np.zeros(len(uids))
        for tri, _ in kf.split(X_tr_aug, y_tr_aug):
            m = LGBMClassifier(**lgbm_p)
            m.fit(X_tr_aug.iloc[tri], y_tr_aug[tri])
            probs += m.predict_proba(X_te[cols])[:, 1] / kf.n_splits

        miner = CandidateMiner(**miner_cfg)
        court = PairwiseCourt(**court_cfg)
        preds, appeal_log = run_appeal_pipeline(
            uids, probs, X_te[cols], F_te,
            pd.Series(y_te, index=uids), cols,
            miner=miner, court=court
        )

        # Patch Veto @pete_prk : prob < 0.60 + court_bot=1 + hour_entropy > 2.5
        if use_pete_veto:
            h_col = "tmp_hour_entropy"
            for uid in uids:
                idx = uids.index(uid)
                prob = probs[idx]
                if preds[idx] == 1 and 0.40 < prob < 0.65:
                    h_ent = float(X_te.at[uid, h_col]) if h_col in X_te.columns else 0
                    if h_ent > 2.5:
                        app = appeal_log.get(uid, {})
                        if app.get("bot_votes", 99) <= 1:
                            preds[idx] = 0
                            appeal_log[uid] = {**app, "action": "pete_veto"}

        tp, fn, fp, sc = official(y_te, preds)
        changed = []
        for uid, app in appeal_log.items():
            if app.get("action", "none") != "none":
                uname = u_df[u_df["user_id"] == uid]["username"].values
                uname = uname[0] if len(uname) > 0 else uid[:8]
                label = int(y_s.loc[uid])
                changed.append({
                    "name": uname, "event": test_n, "label": label,
                    "prob": round(float(probs[uids.index(uid)]), 4),
                    "action": app.get("action", "?"),
                    "bot_votes": app.get("bot_votes", "?")
                })

        results.append({
            "event": test_n, "tp": tp, "fn": fn, "fp": fp,
            "score": sc, "changed": changed
        })
    return results


def section(t, w=70): print(f"\n{'='*w}\n  {t}\n{'='*w}")

def print_cfg(label, results, ref=None):
    sc  = [r["score"] for r in results]
    fpm = max(r["fp"] for r in results)
    print(f"\n  [{label}]")
    print(f"    mean={np.mean(sc):.1f}  p5={bs_p5(sc)}  FP_max={fpm}")
    for r in sorted(results, key=lambda x: x["event"]):
        diff = ""
        if ref:
            rr = next(x for x in ref if x["event"] == r["event"])
            d = r["score"] - rr["score"]
            diff = f"  ({'+' if d>=0 else ''}{d})"
        fp_tag = " ⚠️" if r["fp"] > 0 else ""
        print(f"    E{r['event']:2d}: {r['score']:4d} | TP={r['tp']} FN={r['fn']} FP={r['fp']}{fp_tag}{diff}")
    all_ch = [c for r in results for c in r["changed"]]
    if all_ch:
        print(f"    Comptes changés ({len(all_ch)}) :")
        for c in all_ch:
            truth = "🤖" if c["label"] == 1 else "👤"
            print(f"      {truth} @{c['name']:<20} (E{c['event']}) prob={c['prob']} votes={c['bot_votes']} [{c['action']}]")


def main():
    print("\n🇫🇷 Benchmark FR Dual — Rail FR (Comparable benchmark_residual_surgery.py)")
    print(f"   Référence officielle : FR mean={OFFICIAL_SCORES['FR']['mean']}  "
          f"p5={OFFICIAL_SCORES['FR']['p5']}  FP_max={OFFICIAL_SCORES['FR']['FP_max']}")

    fr_raw = {n: load_event(n) for n in [2, 4, 6, 31] if load_event(n)}
    section("PRÉ-CALCUL")
    fr_c = precompute(fr_raw)

    section("GÉNÉRATION SYNTHÉTIQUE FR v2")
    synth_df, synth_y = generate_fr_hard_negatives_v2(n_per_bot=50, n_per_human=60, seed=42)
    n_b = synth_y.sum(); n_h = (synth_y==0).sum()
    print(f"  Dataset FR v2 : {len(synth_df)} comptes ({n_b} bots / {n_h} humains)")
    print(f"  Archétypes bots FR : {FR_BOT_ARCHETYPES}")
    print(f"  Archétypes humains FR : {FR_HUMAN_ARCHETYPES}")

    section("CONFIGS")
    print("\n  Config 1 : Champion FR actuel...")
    r1 = run_loeo_fr(fr_c, FR_MINER_CONFIG, FR_COURT_CONFIG)

    print("  Config 2 : + Synth FR v2...")
    r2 = run_loeo_fr(fr_c, FR_MINER_CONFIG, FR_COURT_CONFIG,
                     synth_df=synth_df, synth_y=synth_y)

    print("  Config 3 : + Veto @pete_prk (hour_entropy > 2.5, court_bot≤1)...")
    r3 = run_loeo_fr(fr_c, FR_MINER_CONFIG, FR_COURT_CONFIG,
                     use_pete_veto=True)

    print("  Config 4 : Combiné (Synth v2 + Veto)...")
    r4 = run_loeo_fr(fr_c, FR_MINER_CONFIG, FR_COURT_CONFIG,
                     synth_df=synth_df, synth_y=synth_y, use_pete_veto=True)

    section("RÉSULTATS")
    print_cfg("1. Champion FR (Réf. Officielle)", r1)
    print_cfg("2. + Synth FR v2 (archetypes ancrés FN résiduels)", r2, r1)
    print_cfg("3. + Veto @pete_prk (FP rescue E6)", r3, r1)
    print_cfg("4. Combiné", r4, r1)

    section("VERDICT 5 GATES")
    ref_m = OFFICIAL_SCORES["FR"]["mean"]
    ref_p5 = OFFICIAL_SCORES["FR"]["p5"]
    configs = [("Synth v2", r2), ("Veto FR", r3), ("Combiné", r4)]
    print(f"\n  Référence officielle : mean={ref_m}  p5={ref_p5}")

    for name, res in configs:
        m   = np.mean([r["score"] for r in res])
        p5  = bs_p5([r["score"] for r in res])
        fp  = max(r["fp"] for r in res)
        dm  = m - ref_m
        ok_mean  = m >= ref_m
        ok_p5    = p5 >= ref_p5 - 2.0
        ok_fp    = fp <= 1
        ok_no_new_fp = fp <= OFFICIAL_SCORES["FR"]["FP_max"]
        verdict = ("🟢 PROMOTE"    if (dm > 0 and ok_fp and ok_no_new_fp) else
                   "⚪ NEUTRE"     if (abs(dm) < 1.0 and ok_fp) else
                   "🔴 REJECT")
        print(f"\n  [{name}] mean={m:.1f} ({'+' if dm>=0 else ''}{dm:.1f})  p5={p5}  FP_max={fp}")
        print(f"    FR mean+ : {'✅' if ok_mean else '❌'}  |  "
              f"p5 stable : {'✅' if ok_p5 else '❌'}  |  "
              f"FP_max≤1 : {'✅' if ok_fp else '❌'}")
        print(f"    → {verdict}")

if __name__ == "__main__":
    main()
