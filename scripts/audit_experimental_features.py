import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.champion_config import (
    EN_MONOLITH_CONFIG, FR_MONOLITH_CONFIG,
    EN_COURT_CONFIG, FR_COURT_CONFIG,
    EN_MINER_CONFIG, FR_MINER_CONFIG
)
from scripts.residual_error_autopsy import predict_detailed, load_event, get_username

SIM_EN_TEST = 5; SIM_FR_TEST = 6
SIM_EN_TRAIN = [1, 3, 30]; SIM_FR_TRAIN = [2, 4, 31]

ev_test_en = load_event(SIM_EN_TEST)
train_en = [load_event(n) for n in SIM_EN_TRAIN]

ev_test_fr = load_event(SIM_FR_TEST)
train_fr = [load_event(n) for n in SIM_FR_TRAIN]

from src.champion_config import get_fr_synthetic_data
synth_df, synth_y = get_fr_synthetic_data()

TEST_FLAGS = [
    "use_register_invariance",
    "use_vas",
    "use_human_mimicry",
    "use_lrh",
    "use_lrh2",
    "use_lrh3",
    "use_temporal_motifs",
    "use_semantic_coherence",
]

def get_errors(lang, config):
    if lang == "EN":
        preds, probs, log, feat, imp, uids, y_true, *rest = predict_detailed(
            ev_test_en, train_en, EN_MINER_CONFIG, EN_COURT_CONFIG, config, lang="EN"
        )
        u_df = ev_test_en["u"]
    else:
        preds, probs, log, feat, imp, uids, y_true, *rest = predict_detailed(
            ev_test_fr, train_fr, FR_MINER_CONFIG, FR_COURT_CONFIG, config, synth_df, synth_y, lang="FR"
        )
        u_df = ev_test_fr["u"]
        
    errors = {}
    for i, uid in enumerate(uids):
        p, y = preds[i], y_true[i]
        if p != y:
            uname = get_username(u_df, uid)
            etype = "FP" if y == 0 else "FN"
            errors[uid] = {"username": uname, "type": etype}
    return errors

print("Computing BASELINE...")
base_en = get_errors("EN", EN_MONOLITH_CONFIG)
base_fr = get_errors("FR", FR_MONOLITH_CONFIG)

base_fn_en = sum(1 for v in base_en.values() if v["type"]=="FN")
base_fp_en = sum(1 for v in base_en.values() if v["type"]=="FP")
base_fn_fr = sum(1 for v in base_fr.values() if v["type"]=="FN")
base_fp_fr = sum(1 for v in base_fr.values() if v["type"]=="FP")

results = []
for flag in TEST_FLAGS:
    print(f"Testing {flag}...")
    
    cfg_en = EN_MONOLITH_CONFIG.copy()
    cfg_en[flag] = True
    err_en = get_errors("EN", cfg_en)
    
    cfg_fr = FR_MONOLITH_CONFIG.copy()
    cfg_fr[flag] = True
    err_fr = get_errors("FR", cfg_fr)
    
    def diff(base, new):
        rescued_fn = [v["username"] for k,v in base.items() if v["type"]=="FN" and k not in new]
        saved_fp   = [v["username"] for k,v in base.items() if v["type"]=="FP" and k not in new]
        new_fn     = [v["username"] for k,v in new.items() if v["type"]=="FN" and k not in base]
        new_fp     = [v["username"] for k,v in new.items() if v["type"]=="FP" and k not in base]
        return rescued_fn, saved_fp, new_fn, new_fp

    r_fn_e, s_fp_e, n_fn_e, n_fp_e = diff(base_en, err_en)
    r_fn_f, s_fp_f, n_fn_f, n_fp_f = diff(base_fr, err_fr)
    
    score_change_en = (len(r_fn_e)*2 + len(s_fp_e)) - (len(n_fn_e)*2 + len(n_fp_e))
    score_change_fr = (len(r_fn_f)*2 + len(s_fp_f)) - (len(n_fn_f)*2 + len(n_fp_f))
    
    results.append({
        "flag": flag,
        "score_delta_en": score_change_en,
        "score_delta_fr": score_change_fr,
        "EN_res": r_fn_e + s_fp_e,
        "EN_new": n_fn_e + n_fp_e,
        "FR_res": r_fn_f + s_fp_f,
        "FR_new": n_fn_f + n_fp_f
    })

lines = [
    "# Audit Structuré des Features Expérimentales ('Poubelles')",
    "",
    "| Feature | Pts (EN) | Pts (FR) | Succès (Repêchés FP/FN) | Catastrophes (Nouveaux FP/FN) |",
    "| --- | --- | --- | --- | --- |"
]
for r in results:
    s_en = r["score_delta_en"]
    s_fr = r["score_delta_fr"]
    res = ", ".join(r["EN_res"] + r["FR_res"]) or "Aucun"
    new = ", ".join(r["EN_new"] + r["FR_new"]) or "Aucun"
    lines.append(f"| {r['flag']} | {s_en:+} | {s_fr:+} | {res} | {new} |")
    
with open("artifacts/experimental_features_audit.md", "w", encoding="utf-8") as f:
    f.write(chr(10).join(lines))
print("DONE")
