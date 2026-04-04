#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_tournament.py
======================
Feature Tournament — 5 phases disciplinées.

Phase 1 : Inventaire complet de toutes les features et familles
Phase 2 : Test unitaire par famille (ablation LOEO)
Phase 3 : Classification KEEP_STRONG / KEEP_CONDITIONAL / REDUNDANT / HARMFUL
Phase 4 : Combinaisons intelligentes (forward selection restreinte)
Phase 5 : Leaderboard final + rapport

Protocole :
  EN LOEO sur [1, 3, 5, 30]  — metric principale : mean + E5
  FR LOEO sur [2, 4, 6, 31]  — metric principale : mean + E6
  Score : 2*TP - 2*FN - 6*FP
  Champion baseline : mean EN=101.5, mean FR=47.5

Règles :
  - On ne promeut rien sans gain net clair (delta > 0 sur mean ET p5)
  - HARMFUL si E5 ou E6 dégrade AND p5 dégrade
  - REDUNDANT si delta ~ 0 sur toutes métriques
"""

import os, sys, json, warnings, itertools, time
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.candidate_miner_court import CandidateMiner, PairwiseCourt, run_appeal_pipeline
from src.features.forensic_humanness   import extract_forensic_humanness
from src.champion_config import (
    EN_MINER_CONFIG, EN_COURT_CONFIG,
    FR_MINER_CONFIG, FR_COURT_CONFIG,
    MONOLITH_CONFIG, get_lgbm_params, get_kfold_params,
    get_fr_synthetic_data, OFFICIAL_SCORES,
)

# ─── Constantes ──────────────────────────────────────────────────────────────

EN_EVENTS = [1, 3, 5, 30]
FR_EVENTS = [2, 4, 6, 31]
DATASET_DIR = "dataset"
OUT_DIR = Path("artifacts")

CHAMPION_EN = OFFICIAL_SCORES["EN"]["mean"]   # 101.5
CHAMPION_FR = OFFICIAL_SCORES["FR"]["mean"]   # 47.5
CHAMPION_E5 = OFFICIAL_SCORES["EN"]["E5"]     # 92
CHAMPION_E6 = OFFICIAL_SCORES["FR"]["E6"]     # 34

# ─── Feature flags disponibles ───────────────────────────────────────────────
# Structure : { flag_name : { family, description, champion_state, cols } }

ALL_FLAGS = {
    "use_vas": {
        "family": "visual",
        "description": "Visual Attention Spammer — emojis alerte, majuscules, clickbait",
        "champion_state": True,
        "cols": ["vas_attention_ratio","vas_emotion_ratio","vas_emoji_spam_idx",
                 "vas_all_caps_ratio","vas_clickbait_ratio","vas_spam_score"],
        "lang_target": "both",
    },
    "use_lrh": {
        "family": "lrh",
        "description": "LRH — Archétypes legit humains (poll, stan, rp, sports)",
        "champion_state": True,
        "cols": ["lrh_fandom_score","lrh_rp_score","lrh_poll_score","lrh_sports_score",
                 "lrh_promo_score","lrh_intent_score","lrh_specialist_max","lrh_legit_score"],
        "lang_target": "EN",
    },
    "use_lrh2": {
        "family": "lrh",
        "description": "LRH2 — Résiduel poll+rp+stan score",
        "champion_state": True,
        "cols": ["lrh2_poll_score","lrh2_rp_score","lrh2_stan_score","lrh2_residual_score"],
        "lang_target": "EN",
    },
    "use_lrh3": {
        "family": "lrh",
        "description": "LRH3 — Archetype rescue hyper-ciblé (poll topical, rp hardened)",
        "champion_state": True,
        "cols": ["lrh3_poll_topical","lrh3_rp_hardened","lrh3_pun_density",
                 "lrh3_pun_fandom_combo","lrh3_rescue_score"],
        "lang_target": "EN",
    },
    "use_temporal_motifs": {
        "family": "temporal",
        "description": "Temporal Motifs — burst_ratio, n_sessions, sl_ratio",
        "champion_state": False,
        "cols": ["temporal_burst_ratio","temporal_n_sessions","temporal_sl_ratio"],
        "lang_target": "both",
    },
    "use_semantic_coherence": {
        "family": "text",
        "description": "Semantic Coherence — TF-IDF cosine bio<->posts",
        "champion_state": False,
        "cols": ["semantic_overlap"],
        "lang_target": "both",
    },
    "use_content_repetition": {
        "family": "text",
        "description": "Content Repetition — Jaccard consec posts (CSR) [REJETÉ]",
        "champion_state": False,
        "cols": ["csr_jaccard_mean","csr_jaccard_std","csr_jaccard_min",
                 "csr_unigram_pct_shared","csr_template_score"],
        "lang_target": "both",
    },
    "use_high_roi": {
        "family": "temporal",
        "description": "High ROI — time_delta_v2 + sentiment_volatility",
        "champion_state": False,
        "cols": ["td_mean_delta","td_cv_delta","sv_volatility","sv_skew"],
        "lang_target": "both",
    },
    "use_register_invariance": {
        "family": "text",
        "description": "Register Invariance Detector (RID) — register cv, ttr_variance",
        "champion_state": False,
        "cols": ["rid_register_cv","rid_punct_stability","rid_ttr_variance",
                 "rid_interval_regularity","rid_skeleton_entropy","rid_topic_lock","rid_stealth_score"],
        "lang_target": "both",
    },
}

# Always-on feature families (pas de flag)
ALWAYS_ON_FAMILIES = [
    {"family": "base",      "description": "Competition base features (temporal, text, structural)", "cols": "all_base"},
    {"family": "incon",     "description": "Human Inconsistency detector",                          "cols": "incon_*"},
    {"family": "hacker",    "description": "Hacker pipeline (clock, phonotactics, madlibs)",       "cols": "clk_*|ph_*|sk_*"},
    {"family": "v2",        "description": "V2 metadata enriched (event_relative, ghost_prot_v2)", "cols": "v2_*"},
    {"family": "ghost_slim","description": "Ghost Slim (validé prod — gh_n_posts, len_cv)",        "cols": "gh_*"},
    {"family": "forensic",  "description": "Forensic humanness (always-on via pipeline)",          "cols": "forensic_*"},
    {"family": "court",     "description": "Pairwise Court (KNN 5-espaces)",                      "cols": "court_*"},
    {"family": "miner",     "description": "Candidate Miner (veto+rescue pipeline)",               "cols": "miner_*"},
]

# ─── Chargement ──────────────────────────────────────────────────────────────

def load_event(n):
    jp = f"{DATASET_DIR}/dataset.posts&users.{n}.json"
    bp = f"{DATASET_DIR}/dataset.bots.{n}.txt"
    with open(jp, encoding="utf-8") as f: d = json.load(f)
    u = pd.DataFrame(d["users"]).rename(columns={"id": "user_id"})
    p = pd.DataFrame(d["posts"]).rename(columns={"author_id": "user_id"})
    u["user_id"] = u["user_id"].astype(str)
    p["user_id"] = p["user_id"].astype(str)
    with open(bp, encoding="utf-8") as f:
        bots = {s.strip() for s in f if s.strip()}
    u["is_bot"] = u["user_id"].isin(bots).astype(int)
    return {"u": u, "p": p, "meta": d.get("metadata", {}), "bots": bots, "n": n}

def extract_feats(ev, config):
    from src.pipeline.monolithic_extractor import extract_monolithic_features
    feat = extract_monolithic_features(ev["u"], ev["p"], ev["meta"], config=config)
    feat_n = feat.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
    foren  = extract_forensic_humanness(ev["u"], ev["p"]).set_index("user_id")
    labels = ev["u"].set_index("user_id")["is_bot"]
    return feat_n, foren, labels

def official_score(y_true, y_pred):
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    return tp, fn, fp, 2*tp - 2*fn - 6*fp


# ─── LOEO benchmark ──────────────────────────────────────────────────────────

def run_loeo(lang, event_ids, config, miner_cfg, court_cfg,
             synth_df=None, synth_y=None):
    """
    Leave-One-Event-Out benchmark.
    Pour chaque event n dans event_ids :
      - train sur tous les autres
      - test sur n
    Retourne dict : {n: score, "mean": mean, "p5": p5}
    """
    lgbm_p = get_lgbm_params(); kf_p = get_kfold_params()
    results = {}

    for test_n in event_ids:
        train_ns = [x for x in event_ids if x != test_n]
        test_ev  = load_event(test_n)
        feat_te, foren_te, labels_te = extract_feats(test_ev, config)
        uids = list(feat_te.index)
        y_true = labels_te.loc[uids].values

        trains = [load_event(n) for n in train_ns]
        X_tr = pd.concat([extract_feats(ev, config)[0] for ev in trains])
        y_tr = np.concatenate([extract_feats(ev, config)[2].values for ev in trains])
        cols  = [c for c in X_tr.columns if c in feat_te.columns]

        if synth_df is not None and lang == "FR":
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
            probs += m.predict_proba(feat_te[cols])[:, 1] / kf.n_splits

        y_dummy = pd.Series(np.zeros(len(uids)), index=uids)
        miner   = CandidateMiner(**miner_cfg)
        court   = PairwiseCourt(**court_cfg)
        preds, _ = run_appeal_pipeline(
            uids, probs, feat_te[cols], foren_te, y_dummy, cols,
            miner=miner, court=court
        )
        _, _, _, sc = official_score(y_true, np.array(preds))
        results[f"E{test_n}"] = sc

    scores = list(results.values())
    results["mean"] = round(np.mean(scores), 1)
    results["p5"]   = round(np.percentile(scores, 5), 1)
    return results


# ─── Ablation runner ─────────────────────────────────────────────────────────

def make_config_variant(base_config, flag, state):
    c = deepcopy(base_config)
    c[flag] = state
    return c

def run_ablation(flag, base_config, lang, event_ids, miner_cfg, court_cfg,
                 synth_df=None, synth_y=None):
    """
    Teste le flag ON et OFF vs la config de base.
    Retourne { "on": results, "off": results }
    """
    current_state = base_config.get(flag, False)

    # Test ON
    cfg_on  = make_config_variant(base_config, flag, True)
    r_on    = run_loeo(lang, event_ids, cfg_on, miner_cfg, court_cfg, synth_df, synth_y)

    # Test OFF
    cfg_off = make_config_variant(base_config, flag, False)
    r_off   = run_loeo(lang, event_ids, cfg_off, miner_cfg, court_cfg, synth_df, synth_y)

    return {"on": r_on, "off": r_off, "champion_state": current_state}


# ─── Classify feature ─────────────────────────────────────────────────────────

def classify_feature(flag, abl_en, abl_fr):
    """Classe une feature sur la base de l'ablation EN+FR."""
    champ_state = ALL_FLAGS[flag]["champion_state"]

    # Delta = ON - OFF (gain d'activer la feature)
    d_en_mean = abl_en["on"]["mean"] - abl_en["off"]["mean"]
    d_fr_mean = abl_fr["on"]["mean"] - abl_fr["off"]["mean"] if abl_fr else 0.0

    helps_en = d_en_mean > 0.5
    helps_fr = d_fr_mean > 0.5
    hurts_en = d_en_mean < -1.0
    hurts_fr = d_fr_mean < -1.0

    if hurts_en or hurts_fr:
        verdict = "HARMFUL"
    elif abs(d_en_mean) < 0.5 and abs(d_fr_mean) < 0.5:
        verdict = "REDUNDANT"
    elif helps_en and helps_fr:
        verdict = "KEEP_STRONG"
    elif helps_en or helps_fr:
        verdict = "KEEP_CONDITIONAL"
    else:
        verdict = "REDUNDANT"

    return {
        "flag": flag,
        "family": ALL_FLAGS[flag]["family"],
        "champion_state": champ_state,
        "lang_target": ALL_FLAGS[flag]["lang_target"],
        "d_en_mean": round(d_en_mean, 1),
        "d_fr_mean": round(d_fr_mean, 1),
        "helps_EN": helps_en,
        "helps_FR": helps_fr,
        "hurts_EN": hurts_en,
        "hurts_FR": hurts_fr,
        "verdict": verdict,
        "description": ALL_FLAGS[flag]["description"],
    }


# ─── Phase 4 : Forward selection ─────────────────────────────────────────────

def forward_selection(lang, base_config, event_ids, miner_cfg, court_cfg,
                      candidate_flags, baseline_mean, synth_df=None, synth_y=None,
                      max_steps=4):
    """
    Forward selection restreinte.
    Commence depuis base_config, ajoute une feature à la fois si delta > 0.
    Ne teste que les candidate_flags fournis.
    """
    current_config = deepcopy(base_config)
    current_mean   = baseline_mean
    history        = []
    selected       = []

    for step in range(max_steps):
        best_flag   = None
        best_delta  = 0.0
        best_result = None

        for flag in candidate_flags:
            if flag in selected:
                continue
            cfg = make_config_variant(current_config, flag, True)
            r   = run_loeo(lang, event_ids, cfg, miner_cfg, court_cfg, synth_df, synth_y)
            delta = r["mean"] - current_mean
            if delta > best_delta:
                best_delta  = delta
                best_flag   = flag
                best_result = r

        if best_flag is None or best_delta <= 0:
            break  # rien de mieux — stop

        current_config = make_config_variant(current_config, best_flag, True)
        current_mean   = best_result["mean"]
        selected.append(best_flag)
        history.append({
            "step":     step + 1,
            "added":    best_flag,
            "new_mean": best_result["mean"],
            "delta":    round(best_delta, 1),
            "result":   best_result,
        })
        print(f"    [{lang} step {step+1}] +{best_flag}: {round(best_delta,1):+.1f} → mean={current_mean}")

    return selected, history, current_config, current_mean


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_feature_inventory():
    """Phase 1 — Inventaire statique complet."""
    rows = []

    # Features avec flag (activables/désactivables)
    for flag, info in ALL_FLAGS.items():
        active_en = info["champion_state"]
        active_fr = info["champion_state"]
        rows.append({
            "flag":            flag,
            "family":          info["family"],
            "description":     info["description"],
            "active_in_champion": info["champion_state"],
            "lang_target":     info["lang_target"],
            "n_cols":          len(info["cols"]),
            "cols":            " | ".join(info["cols"]),
            "component":       "monolith_extractor",
            "type":            "flagged",
            "status":          "ACTIVE" if info["champion_state"] else "INACTIVE",
        })

    # Features toujours actives
    always_on = [
        {"flag":"base_competition",    "family":"base",     "description":"Competition base (temporal, text, usr, struct)", "cols":"all", "lang_target":"both"},
        {"flag":"human_inconsistency", "family":"incon",    "description":"Human inconsistency signals",                    "cols":"incon_*","lang_target":"both"},
        {"flag":"clock_forensics",     "family":"hacker",   "description":"Clock forensics (posting hours)",               "cols":"clk_*","lang_target":"both"},
        {"flag":"username_phonotactics","family":"hacker",  "description":"Username phonotactics (digit ratio, etc.)",     "cols":"ph_*","lang_target":"both"},
        {"flag":"template_madlibs",    "family":"hacker",   "description":"Template madlibs (skeleton repetition)",        "cols":"sk_*","lang_target":"both"},
        {"flag":"event_relative_v2",   "family":"v2",       "description":"Event-relative normalization v2",               "cols":"v2_rel_*","lang_target":"both"},
        {"flag":"contradiction_v2",    "family":"v2",       "description":"Contradiction v2 (metadata)",                   "cols":"v2_contra_*","lang_target":"both"},
        {"flag":"ghost_protector_v2",  "family":"v2",       "description":"Ghost protector v2 (v2_*)",                     "cols":"v2_ghost_*","lang_target":"both"},
        {"flag":"ghost_slim",          "family":"ghost",    "description":"Ghost Slim (gh_n_posts, len_cv, skeleton)",     "cols":"gh_*","lang_target":"both"},
        {"flag":"forensic_humanness",  "family":"forensic", "description":"Forensic humanness score (always-on)",          "cols":"forensic_*","lang_target":"both"},
        {"flag":"candidate_miner",     "family":"miner",    "description":"Pairwise Court + Candidate Miner",              "cols":"n/a","lang_target":"both"},
    ]
    for ao in always_on:
        rows.append({
            "flag":            ao["flag"],
            "family":          ao["family"],
            "description":     ao["description"],
            "active_in_champion": True,
            "lang_target":     ao["lang_target"],
            "n_cols":          "?",
            "cols":            ao["cols"],
            "component":       "monolith_extractor",
            "type":            "always_on",
            "status":          "ACTIVE",
        })

    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("\n" + "═"*70)
    print("  🏟  FEATURE TOURNAMENT — 5 phases disciplinées")
    print(f"  Champions : EN={CHAMPION_EN} | FR={CHAMPION_FR}")
    print("═"*70)

    # ── Chargement synth FR
    print("\n  Synth FR v2 ...")
    synth_df, synth_y = get_fr_synthetic_data()

    # ── PHASE 1 : Inventaire
    print("\n─── PHASE 1 : Inventaire ───")
    inv_df = build_feature_inventory()
    inv_df.to_csv(OUT_DIR / "feature_inventory.csv", index=False, encoding="utf-8")

    inv_md = ["# 📋 Feature Inventory\n"]
    inv_md.append(f"Total : {len(inv_df)} features/groupes\n")
    inv_md.append("## Features flaguées (activables)")
    flagged = inv_df[inv_df["type"]=="flagged"]
    for _, row in flagged.iterrows():
        status = "✅ ON" if row["active_in_champion"] else "⬜ OFF"
        inv_md.append(f"- `{row['flag']}` | {row['family']} | {status} | {row['lang_target']} | {row['description']}")
    inv_md.append("\n## Features always-on")
    always = inv_df[inv_df["type"]=="always_on"]
    for _, row in always.iterrows():
        inv_md.append(f"- `{row['flag']}` | {row['family']} | ✅ toujours actif")
    (OUT_DIR / "feature_inventory.md").write_text("\n".join(inv_md), encoding="utf-8")
    print(f"  ✓ Inventaire : {len(inv_df)} entrées")

    # ── PHASE 2 : Ablation unitaire par flag
    print("\n─── PHASE 2 : Ablation unitaire ───")
    ablation_results = {}

    for flag, info in ALL_FLAGS.items():
        desc = info["description"][:40]
        print(f"\n  [{flag}] {desc}")

        # EN ablation
        print(f"    EN ablation ...")
        abl_en = run_ablation(flag, MONOLITH_CONFIG, "EN", EN_EVENTS,
                              EN_MINER_CONFIG, EN_COURT_CONFIG)
        print(f"    EN ON={abl_en['on']['mean']:.1f} OFF={abl_en['off']['mean']:.1f} "
              f"delta={abl_en['on']['mean']-abl_en['off']['mean']:+.1f}")

        # FR ablation
        print(f"    FR ablation ...")
        abl_fr = run_ablation(flag, MONOLITH_CONFIG, "FR", FR_EVENTS,
                              FR_MINER_CONFIG, FR_COURT_CONFIG,
                              synth_df=synth_df, synth_y=synth_y)
        print(f"    FR ON={abl_fr['on']['mean']:.1f} OFF={abl_fr['off']['mean']:.1f} "
              f"delta={abl_fr['on']['mean']-abl_fr['off']['mean']:+.1f}")

        ablation_results[flag] = {"en": abl_en, "fr": abl_fr}

    # ── PHASE 3 : Classification
    print("\n─── PHASE 3 : Classification ───")
    classified = []
    for flag, abl in ablation_results.items():
        cls = classify_feature(flag, abl["en"], abl["fr"])
        # Ajouter scores event-level
        cls["EN_on_mean"]  = abl["en"]["on"]["mean"]
        cls["EN_off_mean"] = abl["en"]["off"]["mean"]
        cls["FR_on_mean"]  = abl["fr"]["on"]["mean"]
        cls["FR_off_mean"] = abl["fr"]["off"]["mean"]
        cls["EN_on_E5"]    = abl["en"]["on"].get("E5", abl["en"]["on"].get("E30", 0))
        cls["FR_on_E6"]    = abl["fr"]["on"].get("E6", abl["fr"]["on"].get("E31", 0))
        classified.append(cls)
        print(f"  {flag:<30} → {cls['verdict']} (ΔEN={cls['d_en_mean']:+.1f} ΔFR={cls['d_fr_mean']:+.1f})")

    cls_df = pd.DataFrame(classified)

    # Build unitary report
    lines = ["# 🧪 Feature Unitary Report\n"]
    lines.append(f"**Champion baseline** : EN mean={CHAMPION_EN} | FR mean={CHAMPION_FR}\n")
    lines.append("| Feature | Family | Champion | ΔEN mean | ΔFR mean | E5 ON | E6 ON | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in cls_df.sort_values("verdict").iterrows():
        champ = "✅" if row["champion_state"] else "⬜"
        en_e5 = f"{row['EN_on_E5']:.0f}"
        fr_e6 = f"{row['FR_on_E6']:.0f}"
        lines.append(f"| `{row['flag']}` | {row['family']} | {champ} | {row['d_en_mean']:+.1f} | {row['d_fr_mean']:+.1f} | {en_e5} | {fr_e6} | **{row['verdict']}** |")

    lines.append("\n## Détail par verdict\n")
    for verdict in ["KEEP_STRONG","KEEP_CONDITIONAL","REDUNDANT","HARMFUL"]:
        subset = cls_df[cls_df["verdict"]==verdict]
        lines.append(f"### {verdict} ({len(subset)})")
        for _, row in subset.iterrows():
            c = "✅" if row["champion_state"] else "⬜"
            lines.append(f"- `{row['flag']}` {c} — {row['description']}")
            lines.append(f"  ΔEN={row['d_en_mean']:+.1f} ΔFR={row['d_fr_mean']:+.1f} | helps_EN={row['helps_EN']} helps_FR={row['helps_FR']}")
        lines.append("")

    (OUT_DIR / "feature_unitary_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Family ablation report
    fam_lines = ["# 👨‍👩‍👧 Family Ablation Report\n"]
    family_stats = cls_df.groupby("family").agg(
        n_features=("flag","count"),
        mean_d_en=("d_en_mean","mean"),
        mean_d_fr=("d_fr_mean","mean"),
        keep_strong=("verdict",lambda x:(x=="KEEP_STRONG").sum()),
        harmful=("verdict",lambda x:(x=="HARMFUL").sum()),
    ).reset_index()

    fam_lines.append("| Family | N features | ΔEN avg | ΔFR avg | KEEP_STRONG | HARMFUL |")
    fam_lines.append("|---|---|---|---|---|---|")
    for _, row in family_stats.sort_values("mean_d_en",ascending=False).iterrows():
        fam_lines.append(f"| {row['family']} | {row['n_features']} | {row['mean_d_en']:+.1f} | {row['mean_d_fr']:+.1f} | {row['keep_strong']} | {row['harmful']} |")

    fam_lines.append("\n### Always-on families (non ablatable)\n")
    for ao in ALWAYS_ON_FAMILIES:
        fam_lines.append(f"- **{ao['family']}** : {ao['description']}")

    (OUT_DIR / "family_ablation_report.md").write_text("\n".join(fam_lines), encoding="utf-8")
    print("  ✓ Unitary report + Family report générés")

    # ── PHASE 4 : Forward selection
    print("\n─── PHASE 4 : Forward Selection ───")

    # Candidats = features KEEP_STRONG ou KEEP_CONDITIONAL
    en_candidates = cls_df[cls_df["verdict"].isin(["KEEP_STRONG","KEEP_CONDITIONAL"]) &
                            cls_df["helps_EN"]]["flag"].tolist()
    fr_candidates = cls_df[cls_df["verdict"].isin(["KEEP_STRONG","KEEP_CONDITIONAL"]) &
                            cls_df["helps_FR"]]["flag"].tolist()

    # Baseline = champion (flags déjà actifs par champion_config)
    baseline_en_r = run_loeo("EN", EN_EVENTS, MONOLITH_CONFIG, EN_MINER_CONFIG, EN_COURT_CONFIG)
    baseline_fr_r = run_loeo("FR", FR_EVENTS, MONOLITH_CONFIG, FR_MINER_CONFIG, FR_COURT_CONFIG,
                              synth_df=synth_df, synth_y=synth_y)
    baseline_en  = baseline_en_r["mean"]
    baseline_fr  = baseline_fr_r["mean"]
    print(f"  Baseline EN: {baseline_en:.1f} | FR: {baseline_fr:.1f}")

    leaderboard = []
    # Entry baseline
    leaderboard.append({
        "candidate": "champion_baseline",
        "features_added": "",
        "EN_mean": baseline_en,
        "FR_mean": baseline_fr,
        "total_proxy": round(baseline_en + baseline_fr, 1),
        "EN_E5": baseline_en_r.get("E5", 0),
        "FR_E6": baseline_fr_r.get("E6", 0),
        "verdict": "BASELINE",
    })

    # Forward selection EN
    print(f"\n  EN forward selection (candidates: {en_candidates})")
    if en_candidates:
        sel_en, hist_en, cfg_en, best_en = forward_selection(
            "EN", MONOLITH_CONFIG, EN_EVENTS, EN_MINER_CONFIG, EN_COURT_CONFIG,
            en_candidates, baseline_en, max_steps=4
        )
        if sel_en:
            r_en_best = run_loeo("EN", EN_EVENTS, cfg_en, EN_MINER_CONFIG, EN_COURT_CONFIG)
            leaderboard.append({
                "candidate": f"EN_forward_+{'+'.join(sel_en)}",
                "features_added": " + ".join(sel_en),
                "EN_mean": r_en_best["mean"],
                "FR_mean": baseline_fr,
                "total_proxy": round(r_en_best["mean"] + baseline_fr, 1),
                "EN_E5": r_en_best.get("E5", 0),
                "FR_E6": baseline_fr_r.get("E6", 0),
                "verdict": "promote" if r_en_best["mean"] > baseline_en else "reject",
            })
    else:
        print("    Aucun candidat EN KEEP_STRONG/CONDITIONAL")

    # Forward selection FR
    print(f"\n  FR forward selection (candidates: {fr_candidates})")
    if fr_candidates:
        sel_fr, hist_fr, cfg_fr, best_fr = forward_selection(
            "FR", MONOLITH_CONFIG, FR_EVENTS, FR_MINER_CONFIG, FR_COURT_CONFIG,
            fr_candidates, baseline_fr, synth_df=synth_df, synth_y=synth_y, max_steps=4
        )
        if sel_fr:
            r_fr_best = run_loeo("FR", FR_EVENTS, cfg_fr, FR_MINER_CONFIG, FR_COURT_CONFIG,
                                  synth_df=synth_df, synth_y=synth_y)
            leaderboard.append({
                "candidate": f"FR_forward_+{'+'.join(sel_fr)}",
                "features_added": " + ".join(sel_fr),
                "EN_mean": baseline_en,
                "FR_mean": r_fr_best["mean"],
                "total_proxy": round(baseline_en + r_fr_best["mean"], 1),
                "EN_E5": baseline_en_r.get("E5", 0),
                "FR_E6": r_fr_best.get("E6", 0),
                "verdict": "promote" if r_fr_best["mean"] > baseline_fr else "reject",
            })
    else:
        print("    Aucun candidat FR KEEP_STRONG/CONDITIONAL")

    # ── PHASE 5 : Leaderboard final
    print("\n─── PHASE 5 : Leaderboard final ───")
    lb_df = pd.DataFrame(leaderboard).sort_values("total_proxy", ascending=False)
    lb_df.to_csv(OUT_DIR / "feature_tournament_leaderboard.csv", index=False, encoding="utf-8")

    # Final report
    report_lines = ["# 🏆 Final Feature Combination Report\n"]
    report_lines.append(f"**Durée totale** : {(time.time()-t0)/60:.1f} min\n")
    report_lines.append(f"**Baseline EN** : {baseline_en:.1f} | **Baseline FR** : {baseline_fr:.1f}\n")
    report_lines.append("## Leaderboard\n")
    report_lines.append("| Candidate | Features | EN mean | FR mean | Total | E5 | E6 | Verdict |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in lb_df.iterrows():
        v = "🏆" if row["verdict"]=="promote" else ("📍" if row["verdict"]=="BASELINE" else "❌")
        report_lines.append(f"| {row['candidate']} | {row['features_added'] or '—'} | {row['EN_mean']:.1f} | {row['FR_mean']:.1f} | {row['total_proxy']:.1f} | {row['EN_E5']} | {row['FR_E6']} | {v} {row['verdict']} |")

    report_lines.append("\n## Classification features\n")
    for verdict in ["KEEP_STRONG","KEEP_CONDITIONAL","REDUNDANT","HARMFUL"]:
        subset = cls_df[cls_df["verdict"]==verdict]
        report_lines.append(f"### {verdict} ({len(subset)})")
        for _, row in subset.iterrows():
            c = "✅" if row["champion_state"] else "⬜"
            report_lines.append(f"- `{row['flag']}` {c} ΔEN={row['d_en_mean']:+.1f} ΔFR={row['d_fr_mean']:+.1f}")
        report_lines.append("")

    report_lines.append("## Recommandations finales\n")
    to_promote = lb_df[lb_df["verdict"]=="promote"]
    if len(to_promote):
        report_lines.append("### À promouvoir")
        for _, row in to_promote.iterrows():
            delta_en = row["EN_mean"] - baseline_en
            delta_fr = row["FR_mean"] - baseline_fr
            report_lines.append(f"- **{row['candidate']}** : ΔEN={delta_en:+.1f} ΔFR={delta_fr:+.1f} total={row['total_proxy']:.1f}")
    else:
        report_lines.append("### Aucune combinaison à promouvoir — champion actuel reste optimal.")

    report_lines.append(f"\n*Généré le 2026-04-04 — Feature Tournament terminé*")
    (OUT_DIR / "final_feature_combination_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n{'═'*70}")
    print(f"  ✅ Feature Tournament terminé en {(time.time()-t0)/60:.1f} min")
    print(f"{'═'*70}")
    print(f"  Fichiers générés :")
    print(f"    artifacts/feature_inventory.csv")
    print(f"    artifacts/feature_inventory.md")
    print(f"    artifacts/feature_unitary_report.md")
    print(f"    artifacts/family_ablation_report.md")
    print(f"    artifacts/feature_tournament_leaderboard.csv")
    print(f"    artifacts/final_feature_combination_report.md")
    print()


if __name__ == "__main__":
    main()
