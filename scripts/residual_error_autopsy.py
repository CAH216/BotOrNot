#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
residual_error_autopsy.py
==========================
Autopsie froide de toutes les erreurs résiduelles EN + FR.

Protocole identique à score_competition_real.py :
  EN : train E1+E3+E30 → test E5
  FR : train E2+E4+E31+Synth → test E6

Pour chaque FP et FN, extrait :
  - profil feature complet
  - top 3 features accusatrices / protectrices
  - détail votes Court
  - assignation archétype + root_cause

Génère :
  artifacts/residual_error_root_cause_table.csv
  artifacts/residual_error_root_cause_report.md
"""

import os, sys, json, warnings
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
    EN_MONOLITH_CONFIG, FR_MONOLITH_CONFIG,
    get_lgbm_params, get_kfold_params,
    get_fr_synthetic_data,
)

SIM_EN_TEST  = 5
SIM_FR_TEST  = 6
SIM_EN_TRAIN = [1, 3, 30]
SIM_FR_TRAIN = [2, 4, 31]
DATASET_DIR  = "dataset"
OUT_DIR      = Path("artifacts")


# ─── Chargement ───────────────────────────────────────────────────────────────

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

def extract_features(ev, config):
    feat   = extract_monolithic_features(ev["u"], ev["p"], ev["meta"], config=config)
    feat_n = feat.set_index("user_id").select_dtypes(include=[np.number]).fillna(0)
    foren  = extract_forensic_humanness(ev["u"], ev["p"]).set_index("user_id")
    labels = ev["u"].set_index("user_id")["is_bot"]
    return {"feat": feat_n, "forensic": foren, "labels": labels}

def get_username(u_df, uid):
    row = u_df[u_df["user_id"] == uid]
    if not row.empty and "username" in row.columns:
        return str(row["username"].values[0])
    return uid[:10]
def get_field(u_df, uid, field, default="?"):
    row = u_df[u_df["user_id"] == uid]
    if not row.empty and field in row.columns:
        v = row[field].values[0]
        return str(v) if pd.notna(v) else default
    return default


# ─── Prédiction avec collecte détaillée ──────────────────────────────────────

def predict_detailed(test_ev, train_list, miner_cfg, court_cfg, monolith_cfg,
                     synth_df=None, synth_y=None, lang="EN"):
    """Retourne preds, probs, appeal_log, feat_df, importances, uids, y_true."""
    lgbm_p = get_lgbm_params(); kf_p = get_kfold_params()
    fdata  = extract_features(test_ev, monolith_cfg)
    feat_te, foren_te, labels = fdata["feat"], fdata["forensic"], fdata["labels"]
    uids    = list(feat_te.index)
    y_true  = labels.loc[uids].values

    trains  = [extract_features(ev, monolith_cfg) for ev in train_list]
    X_tr    = pd.concat([t["feat"] for t in trains])
    y_tr    = np.concatenate([t["labels"].values for t in trains])
    cols    = [c for c in X_tr.columns if c in feat_te.columns]

    if synth_df is not None:
        sc = [c for c in cols if c in synth_df.columns]
        if sc:
            X_sa = pd.DataFrame(0.0, index=synth_df.index, columns=cols)
            for c in sc: X_sa[c] = synth_df[c].values
            X_tr = pd.concat([X_tr[cols], X_sa[cols]])
            y_tr = np.concatenate([y_tr, synth_y])

    kf    = StratifiedKFold(**kf_p)
    probs = np.zeros(len(uids))
    importances = np.zeros(len(cols))

    for tri, _ in kf.split(X_tr[cols], y_tr):
        m = LGBMClassifier(**lgbm_p)
        m.fit(X_tr[cols].iloc[tri], y_tr[tri])
        probs += m.predict_proba(feat_te[cols])[:, 1] / kf.n_splits
        importances += m.feature_importances_ / kf.n_splits

    feat_importance = dict(zip(cols, importances))

    y_dummy   = pd.Series(np.zeros(len(uids)), index=uids)
    miner     = CandidateMiner(**miner_cfg)
    court     = PairwiseCourt(**court_cfg)
    preds, appeal_log = run_appeal_pipeline(
        uids, probs, feat_te[cols], foren_te, y_dummy, cols,
        miner=miner, court=court
    )

    return (np.array(preds), dict(zip(uids, probs)),
            appeal_log, feat_te[cols], feat_importance, uids, y_true,
            foren_te, test_ev["u"], test_ev["p"])


# ─── Analyse feature pour un compte ──────────────────────────────────────────

def get_top_features(uid, feat_df, feat_importance, n=3):
    """Retourne les top N features accusatrices (vers bot) et protectrices (vers human)."""
    if uid not in feat_df.index:
        return [], []
    row = feat_df.loc[uid]
    # Pondérer les valeurs brutes par l'importance LGBM
    scores = {col: float(row[col]) * float(feat_importance.get(col, 0))
              for col in feat_df.columns if col in feat_importance}
    sorted_pos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    sorted_neg = sorted(scores.items(), key=lambda x: x[1])
    top_acc = [f"{k}={v:.3f}" for k,v in sorted_pos[:n] if v > 0]
    top_pro = [f"{k}={v:.3f}" for k,v in sorted_neg[:n] if v < 0]
    return top_acc, top_pro


# ─── Archétype ────────────────────────────────────────────────────────────────

def assign_archetype(uid, u_df, p_df, feat_df, probs_map, is_bot, lang):
    """Assigne un archétype court basé sur les features disponibles."""
    prob = probs_map.get(uid, 0.5)

    # Tenter de récupérer quelques features clés
    def f(col, default=0.0):
        if uid in feat_df.index and col in feat_df.columns:
            v = feat_df.at[uid, col]
            return float(v) if pd.notna(v) else default
        return default

    poll_sc    = f("lrh_poll_score")
    tmp_cv     = f("tmp_ipt_cv")
    lrh2       = f("lrh2_residual_score")
    usr_digits = f("usr_digit_ratio")
    txt_entropy= f("txt_lexical_diversity")
    vas_score  = f("vas_visual_spammer_score")
    rel_reply  = f("rel_reply_diversity")

    # FP archetypes (humains accusés)
    if not is_bot:
        if poll_sc > 0.5:
            return "poll_human"
        if tmp_cv > 1.5:
            return "sports_fan_human"  # haute variance temporelle → bursts sportifs
        if lang == "FR" and prob > 0.8:
            return "insomniac_human_fr"
        if lrh2 > 0.15:
            return "stan_human"
        return "genuinely_ambiguous"

    # FN archetypes (bots manqués)
    if is_bot:
        if prob < 0.2 and txt_entropy > 0.6:
            return "french_clean_persona_bot" if lang == "FR" else "narrative_clean_bot"
        if poll_sc > 0.3 and lang == "FR":
            return "promo_masked_bot"
        if tmp_cv < 0.5 and prob < 0.4:
            return "lifestyle_sleeper_bot"
        if prob > 0.7:
            return "genuinely_ambiguous"
        if lang == "FR":
            return "french_clean_persona_bot"
        return "narrative_clean_bot"

    return "genuinely_ambiguous"


# ─── Root Cause ───────────────────────────────────────────────────────────────

def assign_root_cause(uid, probs_map, appeal_log, feat_df, is_bot, error_type,
                      miner_cfg, archetype):
    """Assigne root_cause + secondary_cause."""
    prob      = probs_map.get(uid, 0.5)
    app       = appeal_log.get(uid, {})
    action    = app.get("action", "none")
    bot_votes = app.get("bot_votes", 0)
    hum_votes = app.get("human_votes", 0)
    nominated = uid in appeal_log

    def f(col, default=0.0):
        if uid in feat_df.index and col in feat_df.columns:
            v = feat_df.at[uid, col]
            return float(v) if pd.notna(v) else default
        return default

    lrh2   = f("lrh2_residual_score")
    poll   = f("lrh_poll_score")
    tmp_cv = f("tmp_ipt_cv")

    primary = "genuinely_indistinguishable"
    secondary = ""

    if error_type == "FP":
        # Humain classé bot
        if action == "none" and prob > 0.8:
            # Le miner n'a pas nominé alors que le modèle était très confiant
            if poll > 0.5 or tmp_cv > 1.5:
                primary   = "temporal_overweight"
                secondary = "miner_not_triggered"
            elif lrh2 > 0.15:
                primary   = "lrh_overweight"
            else:
                primary   = "genuinely_indistinguishable"
        elif action == "veto":
            # Nominé et vetoé → correct comportement
            primary = "court_not_strong_enough"
        else:
            primary = "genuinely_indistinguishable"

    elif error_type == "FN":
        # Bot classé humain
        if prob < 0.3:
            # Monolithe complètement aveugle
            if f("txt_lexical_diversity") > 0.55:
                primary = "genuinely_indistinguishable"
            else:
                primary = "temporal_overweight"
                secondary = "miner_not_triggered"
        elif 0.3 <= prob < 0.5:
            # Prob basse — limite du modèle
            primary = "miner_not_triggered"
            if bot_votes >= 1:
                secondary = "court_not_strong_enough"
        elif prob >= 0.5 and action == "veto":
            # Prob élevée mais court a vetoé
            primary   = "court_not_strong_enough"
            secondary = "miner_not_triggered"
        elif prob >= 0.97 and action == "veto":
            # Bug garde-fou (normalement corrigé maintenant)
            primary   = "court_not_strong_enough"
        else:
            primary = "genuinely_indistinguishable"

    return primary, secondary


# ─── Main autopsy ─────────────────────────────────────────────────────────────

def run_autopsy(lang, test_n, train_ns, miner_cfg, court_cfg, monolith_cfg,
                synth_df=None, synth_y=None):

    print(f"\n{'='*60}\n  Autopsie {lang} — E{test_n}\n{'='*60}")

    test_ev    = load_event(test_n)
    train_list = [load_event(n) for n in train_ns]

    (preds, probs_map, appeal_log,
     feat_df, feat_imp, uids, y_true,
     foren_df, u_df, p_df) = predict_detailed(
        test_ev, train_list, miner_cfg, court_cfg, monolith_cfg, synth_df, synth_y, lang
    )

    errors = []
    for uid, yp, yt in zip(uids, preds, y_true):
        if yp == yt:
            continue  # correct — ignorer

        error_type = "FP" if (yp == 1 and yt == 0) else "FN"
        prob       = probs_map.get(uid, 0.0)
        app        = appeal_log.get(uid, {})
        action     = app.get("action", "none")
        bot_votes  = app.get("bot_votes", 0)
        hum_votes  = app.get("human_votes", 0)
        nominated  = uid in appeal_log
        is_bot     = (yt == 1)

        uname = get_username(u_df, uid)
        archetype, root_cause, secondary = (
            assign_archetype(uid, u_df, p_df, feat_df, probs_map, is_bot, lang),
            *assign_root_cause(uid, probs_map, appeal_log, feat_df, is_bot, error_type,
                               miner_cfg, "")
        )

        top_acc, top_pro = get_top_features(uid, feat_df, feat_imp, n=3)

        # could_be_saved_by_veto : FP, prob > 0.5, non nominé pour veto
        saved_veto = (error_type == "FP" and prob >= 0.5 and action == "none"
                      and miner_cfg.get("use_veto", False))
        # could_be_saved_by_miner : FN, prob entre 0.3 et 0.49, bot_votes existants
        saved_miner = (error_type == "FN" and 0.25 <= prob <= 0.49 and bot_votes >= 1)

        errors.append({
            "lang":               lang,
            "event":              f"E{test_n}",
            "account_id":         uid,
            "username":           uname,
            "error_type":         error_type,
            "archetype":          archetype,
            "model_proba":        round(prob, 4),
            "top_3_accusers":     " | ".join(top_acc) if top_acc else "n/a",
            "top_3_protectors":   " | ".join(top_pro) if top_pro else "n/a",
            "court_votes_bot":    bot_votes,
            "court_votes_human":  hum_votes,
            "court_action":       action,
            "was_nominated":      "yes" if nominated else "no",
            "could_be_saved_by_veto":  "yes" if saved_veto else "no",
            "could_be_saved_by_miner": "yes" if saved_miner else "no",
            "root_cause":         root_cause,
            "secondary_cause":    secondary,
            "notes":              _notes(uid, prob, action, bot_votes, archetype, error_type, lang),
        })

        print(f"  {error_type} @{uname:<22} prob={prob:.3f} "
              f"arch={archetype} cause={root_cause}")

    return errors


def _notes(uid, prob, action, bot_votes, archetype, etype, lang):
    notes = []
    if etype == "FN" and prob > 0.95 and action == "veto":
        notes.append("Prob extrême vetoed par Court — garde-fou actif")
    if etype == "FP" and prob > 0.98:
        notes.append("Prob quasi-certaine — signal très fort pour bot")
    if archetype == "genuinely_ambiguous":
        notes.append("Compte intrinsèquement ambigu")
    if archetype in ("french_clean_persona_bot", "narrative_clean_bot") and prob < 0.25:
        notes.append("Bot quasi-invisible — léxico diversifié")
    if lang == "FR" and etype == "FN":
        notes.append("FN FR récurrent — corpus FR sous-représenté")
    return "; ".join(notes) if notes else ""


def df_to_md(df):
    """Pure-python DataFrame -> markdown table (no tabulate needed)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def write_report(errors, csv_path, report_path):
    df = pd.DataFrame(errors)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  CSV → {csv_path}")

    en = df[df["lang"] == "EN"]
    fr = df[df["lang"] == "FR"]
    total_fn = len(df[df["error_type"]=="FN"])
    total_fp = len(df[df["error_type"]=="FP"])

    md = []
    md.append("# 🔬 Residual Error Root Cause Table — Autopsie Finale")
    md.append("\n**Protocole** : EN train E1+E3+E30 → test E5 | FR train E2+E4+E31+Synth → test E6")
    md.append(f"\n**Total erreurs** : {len(df)} ({total_fn} FN + {total_fp} FP)")
    md.append("\n---\n")

    # ── Table complète
    md.append("## 1. Table Complète des Erreurs Résiduelles\n")
    cols_show = ["lang","event","username","error_type","archetype",
                 "model_proba","court_votes_bot","court_votes_human",
                 "court_action","was_nominated","root_cause","secondary_cause"]
    md.append(df_to_md(df[cols_show]))

    # ── Top features par compte
    md.append("\n\n### Détail features — Accuseurs / Protecteurs\n")
    for _, row in df.iterrows():
        md.append(f"**@{row['username']}** ({row['error_type']} {row['lang']} | arch: {row['archetype']})")
        md.append(f"  - Prob : {row['model_proba']}")
        md.append(f"  - Top accuseurs  : {row['top_3_accusers']}")
        md.append(f"  - Top protecteurs: {row['top_3_protectors']}")
        if row['notes']:
            md.append(f"  - Notes : {row['notes']}")
        md.append("")

    md.append("---\n")

    # ── Analyse EN
    md.append("## 2. Analyse EN\n")
    _section_analysis(en, md, "EN")

    md.append("---\n")

    # ── Analyse FR
    md.append("## 3. Analyse FR\n")
    _section_analysis(fr, md, "FR")

    md.append("---\n")

    # ── Priorisation
    md.append("## 4. Priorisation — Ce qui est récupérable\n")

    recoverable = df[df["could_be_saved_by_miner"]=="yes"]
    low_hanging = df[(df["root_cause"]=="miner_not_triggered") &
                     (df["model_proba"].between(0.25, 0.49))]
    hard_cases  = df[df["root_cause"]=="genuinely_indistinguishable"]

    md.append("### 🟢 Erreurs potentiellement récupérables (prob borderline, court possible)\n")
    if len(low_hanging):
        md.append(df_to_md(low_hanging[["lang","username","error_type","model_proba","archetype","root_cause"]]))
    else:
        md.append("_Aucune erreur clairement récupérable sans R&D._")

    md.append("\n### 🟡 Erreurs récupérables avec patch ciblé (root cause identifiable)\n")
    patchable = df[(df["root_cause"].isin(["temporal_overweight","lrh_overweight","court_not_strong_enough"])) &
                   (df["root_cause"] != "genuinely_indistinguishable")]
    if len(patchable):
        md.append(df_to_md(patchable[["lang","username","error_type","model_proba","archetype","root_cause","secondary_cause","notes"]]))
    else:
        md.append("_Aucune._")

    md.append("\n### 🔴 Erreurs quasi-irrécupérables (genuinely_indistinguishable)\n")
    if len(hard_cases):
        md.append(df_to_md(hard_cases[["lang","username","error_type","model_proba","archetype","notes"]]))
    else:
        md.append("_Aucune._")

    md.append("\n---\n")
    md.append("## 5. Synthèse finale\n")
    rc_counts = df["root_cause"].value_counts()
    md.append("### Répartition root_cause (toutes langues)\n")
    for rc, cnt in rc_counts.items():
        md.append(f"- **{rc}** : {cnt} compte(s)")

    arch_counts = df["archetype"].value_counts()
    md.append("\n### Répartition archétypes\n")
    for a, cnt in arch_counts.items():
        md.append(f"- {a} : {cnt}")

    md.append(f"\n*Autopsie générée le 2026-04-04 — Pipeline champion gelé*")

    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  Report → {report_path}")


def _section_analysis(df_lang, md, lang):
    fn = df_lang[df_lang["error_type"]=="FN"]
    fp = df_lang[df_lang["error_type"]=="FP"]
    md.append(f"- FP ({lang}) : {len(fp)}  |  FN ({lang}) : {len(fn)}\n")

    if len(fn) > 0:
        md.append(f"**FN {lang} — par root_cause :**")
        for rc, cnt in fn["root_cause"].value_counts().items():
            md.append(f"  - {rc} : {cnt}")
        md.append(f"\n**FN {lang} — par archetype :**")
        for a, cnt in fn["archetype"].value_counts().items():
            md.append(f"  - {a} : {cnt}")
        md.append("")

    if len(fp) > 0:
        md.append(f"**FP {lang} — par root_cause :**")
        for rc, cnt in fp["root_cause"].value_counts().items():
            md.append(f"  - {rc} : {cnt}")
        md.append(f"\n**FP {lang} — par archetype :**")
        for a, cnt in fp["archetype"].value_counts().items():
            md.append(f"  - {a} : {cnt}")
        md.append("")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "█"*60)
    print("  🔬 AUTOPSIE FINALE — Residual Error Root Cause Analysis")
    print("█"*60)

    print("\n  Génération synth FR v2...")
    synth_df, synth_y = get_fr_synthetic_data()

    all_errors = []

    # EN autopsy
    en_errors = run_autopsy(
        "EN", SIM_EN_TEST, SIM_EN_TRAIN,
        EN_MINER_CONFIG, EN_COURT_CONFIG, EN_MONOLITH_CONFIG
    )
    all_errors.extend(en_errors)

    # FR autopsy
    fr_errors = run_autopsy(
        "FR", SIM_FR_TEST, SIM_FR_TRAIN,
        FR_MINER_CONFIG, FR_COURT_CONFIG, FR_MONOLITH_CONFIG,
        synth_df=synth_df, synth_y=synth_y
    )
    all_errors.extend(fr_errors)

    print(f"\n  Total erreurs détectées : {len(all_errors)}")

    csv_path    = OUT_DIR / "residual_error_root_cause_table.csv"
    report_path = OUT_DIR / "residual_error_root_cause_report.md"
    write_report(all_errors, csv_path, report_path)

    print(f"\n  ✅ Autopsie terminée.")


if __name__ == "__main__":
    main()
