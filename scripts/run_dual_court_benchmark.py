import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.champion_config import EN_MONOLITH_CONFIG, FR_MONOLITH_CONFIG, get_lgbm_params, get_kfold_params, get_fr_synthetic_data
from src.features.candidate_miner_court import CandidateMiner, PairwiseCourt, run_appeal_pipeline
from scripts.residual_error_autopsy import load_event, extract_features

EVENTS_EN = [1, 3, 30, 5]
EVENTS_FR = [2, 4, 31, 6]

MODES = ["BASELINE", "MICRO_VETO_OLD", "DUAL_COURT_PASSIVE", "DUAL_COURT_ACTIVE"]

def run_benchmark():
    print("Pre-Loading Datasets. This takes a moment...")
    
    # EN Data
    ev_te_en = load_event(5)
    evs_tr_en = [load_event(x) for x in EVENTS_EN[:-1]]
    te_en_f = extract_features(ev_te_en, EN_MONOLITH_CONFIG)
    tr_en_fs = [extract_features(e, EN_MONOLITH_CONFIG) for e in evs_tr_en]
    
    X_tr_en = pd.concat([t["feat"] for t in tr_en_fs])
    y_tr_en = np.concatenate([t["labels"].values for t in tr_en_fs])
    cols_en = [c for c in X_tr_en.columns if c in te_en_f["feat"].columns and c not in ["is_bot", "account_id"]]
    
    # FR Data
    ev_te_fr = load_event(6)
    evs_tr_fr = [load_event(x) for x in EVENTS_FR[:-1]]
    te_fr_f = extract_features(ev_te_fr, FR_MONOLITH_CONFIG)
    tr_fr_fs = [extract_features(e, FR_MONOLITH_CONFIG) for e in evs_tr_fr]
    
    X_tr_fr = pd.concat([t["feat"] for t in tr_fr_fs])
    y_tr_fr = np.concatenate([t["labels"].values for t in tr_fr_fs])
    
    synth_df, synth_y = get_fr_synthetic_data()
    sc = [c for c in X_tr_fr.columns if c in synth_df.columns]
    if sc:
        X_sa = pd.DataFrame(0.0, index=synth_df.index, columns=X_tr_fr.columns)
        for c in sc: X_sa[c] = synth_df[c].values
        X_tr_fr = pd.concat([X_tr_fr, X_sa], axis=0)
        y_tr_fr = np.concatenate([y_tr_fr, synth_y])
        
    cols_fr = [c for c in X_tr_fr.columns if c in te_fr_f["feat"].columns and c not in ["is_bot", "account_id"]]

    p_df_en = pd.concat([ev_te_en["p"]] + [e["p"] for e in evs_tr_en])
    p_df_fr = pd.concat([ev_te_fr["p"]] + [e["p"] for e in evs_tr_fr])

    # K-Fold Base Probs
    def get_base_probs(X_train, y_train, X_test, cols):
        kf_p = get_kfold_params()
        kf = StratifiedKFold(**kf_p)
        lgbm_p = get_lgbm_params()
        probs_base = np.zeros(len(X_test))
        for tri, _ in kf.split(X_train[cols], y_train):
            m = LGBMClassifier(**lgbm_p)
            m.fit(X_train[cols].iloc[tri], y_train[tri])
            probs_base += m.predict_proba(X_test[cols])[:, 1] / kf.n_splits
        return probs_base

    print("Training Base Models...")
    prob_base_en = get_base_probs(X_tr_en, y_tr_en, te_en_f["feat"], cols_en)
    prob_base_fr = get_base_probs(X_tr_fr, y_tr_fr, te_fr_f["feat"], cols_fr)
    
    # Miners and Courts
    miner = CandidateMiner(proba_low=0.10, proba_high=0.45, forensic_percentile=60, use_veto=True)
    court_en = PairwiseCourt(min_bot_votes=2)
    court_fr = PairwiseCourt(min_bot_votes=2)
    
    uids_en = list(te_en_f["feat"].index)
    y_true_en = te_en_f["labels"].loc[uids_en].values
    
    uids_fr = list(te_fr_f["feat"].index)
    y_true_fr = te_fr_f["labels"].loc[uids_fr].values
    
    results = {}
    
    for mode in MODES:
        print(f"\n==============================")
        print(f"⚙️ Running mode: {mode}")
        print(f"==============================")
        
        args_en = {
            "uids": uids_en, "probs_base": prob_base_en, "feat_df": te_en_f["feat"][cols_en],
            "forensic_df": te_en_f["forensic"], "labels": pd.Series(0, index=uids_en),
            "all_cols": cols_en, "miner": miner, "court": court_en,
            "posts_df": p_df_en if "BASELINE" not in mode else None,
            "train_feat": X_tr_en, "train_labels": pd.Series(y_tr_en, index=X_tr_en.index),
            "arbitration_mode": mode.replace("MICRO_VETO_OLD", "MICRO_V1")
        }
        
        args_fr = {
            "uids": uids_fr, "probs_base": prob_base_fr, "feat_df": te_fr_f["feat"][cols_fr],
            "forensic_df": te_fr_f["forensic"], "labels": pd.Series(0, index=uids_fr),
            "all_cols": cols_fr, "miner": miner, "court": court_fr,
            "posts_df": p_df_fr if "BASELINE" not in mode else None,
            "train_feat": X_tr_fr, "train_labels": pd.Series(y_tr_fr, index=X_tr_fr.index),
            "arbitration_mode": mode.replace("MICRO_VETO_OLD", "MICRO_V1")
        }
        
        pr_en, ov_en = run_appeal_pipeline(**args_en)
        pr_fr, ov_fr = run_appeal_pipeline(**args_fr)
        
        tp_en = np.sum((pr_en == 1) & (y_true_en == 1))
        fn_en = np.sum((pr_en == 0) & (y_true_en == 1))
        fp_en = np.sum((pr_en == 1) & (y_true_en == 0))
        
        tp_fr = np.sum((pr_fr == 1) & (y_true_fr == 1))
        fn_fr = np.sum((pr_fr == 0) & (y_true_fr == 1))
        fp_fr = np.sum((pr_fr == 1) & (y_true_fr == 0))
        
        llm_o_en = {k: v for k, v in ov_en.items() if "dual_court" in str(v.get("override", "")) or "forced" in str(v.get("override", ""))}
        llm_o_fr = {k: v for k, v in ov_fr.items() if "dual_court" in str(v.get("override", "")) or "forced" in str(v.get("override", ""))}
        
        score_en = (tp_en * 2) - (fn_en * 2) - (fp_en * 3)
        score_fr = (tp_fr * 2) - (fn_fr * 2) - (fp_fr * 3)
        off_score = score_en + score_fr
        
        results[mode] = {
            "EN": f"TP={tp_en} FN={fn_en} FP={fp_en}",
            "FR": f"TP={tp_fr} FN={fn_fr} FP={fp_fr}",
            "Score": off_score,
            "Overrides_EN": len(llm_o_en),
            "Overrides_FR": len(llm_o_fr),
            "Detail_EN": llm_o_en,
            "Detail_FR": llm_o_fr
        }
        print(f"EN: {results[mode]['EN']} | FR: {results[mode]['FR']} | Score: {off_score}")

    print("\n\n=============== FINAL BENCHMARK ===============")
    for m, r in results.items():
        print(f"\n[{m}]")
        print(f"   EN: {r['EN']}")
        print(f"   FR: {r['FR']}")
        print(f"   Official Score: {r['Score']}")
        
        if r['Overrides_EN'] > 0 or r['Overrides_FR'] > 0:
            print(f"   >>> Total Overrides: {r['Overrides_EN'] + r['Overrides_FR']}")
            for lang, dets, ev_df in [("EN", r['Detail_EN'], ev_te_en["u"]), ("FR", r['Detail_FR'], ev_te_fr["u"])]:
                for uid, o in dets.items():
                    u_row = ev_df[ev_df["user_id"] == uid]
                    uname = u_row["username"].values[0] if not u_row.empty else uid
                    print(f"     {lang} Muted @{uname}: {o['override']}")

    print("\nVERDICT (RECOMMENDATION): KEEP_EXPERIMENTAL OR REJECT?")
    print("If TP increased and FP/FN dropped, PROMOTE! If no overrides, KEEP_EXPERIMENTAL. If FP spiked, REJECT.")

if __name__ == "__main__":
    run_benchmark()
