import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.champion_config import (
    EN_MONOLITH_CONFIG, FR_MONOLITH_CONFIG,
    get_lgbm_params, get_kfold_params, get_fr_synthetic_data
)
from src.features.candidate_miner_court import CandidateMiner, PairwiseCourt, run_appeal_pipeline
from scripts.residual_error_autopsy import load_event, extract_features

EVENTS_EN = [1, 3, 30, 5]
EVENTS_FR = [2, 4, 31, 6]

def evaluate_fold(lang, test_ev_id, train_ev_ids, config):
    print(f"\nEvaluating {lang} (Holdout Event {test_ev_id})...")
    test_ev = load_event(test_ev_id)
    train_evs = [load_event(x) for x in train_ev_ids]
    
    fdata = extract_features(test_ev, config)
    feat_te, foren_te, labels = fdata["feat"], fdata["forensic"], fdata["labels"]
    
    trains = [extract_features(ev, config) for ev in train_evs]
    X_tr = pd.concat([t["feat"] for t in trains])
    y_tr = np.concatenate([t["labels"].values for t in trains])
    
    # Pour FR, ajouter données synthétiques
    if lang == "FR":
        synth_df, synth_y = get_fr_synthetic_data()
        sc = [c for c in X_tr.columns if c in synth_df.columns]
        if sc:
            X_sa = pd.DataFrame(0.0, index=synth_df.index, columns=X_tr.columns)
            for c in sc: X_sa[c] = synth_df[c].values
            X_tr = pd.concat([X_tr, X_sa], axis=0)
            y_tr = np.concatenate([y_tr, synth_y])
            
    # Construction features pures
    cols = [c for c in X_tr.columns if c in feat_te.columns and c not in ["is_bot", "account_id"]]
    
    # Train K-Fold (comme Monolithic Court Base)
    kf_p = get_kfold_params()
    kf = StratifiedKFold(**kf_p)
    lgbm_p = get_lgbm_params()
    
    uids = list(feat_te.index)
    y_true = labels.loc[uids].values
    probs_base = np.zeros(len(uids))
    
    for tri, _ in kf.split(X_tr[cols], y_tr):
        m = LGBMClassifier(**lgbm_p)
        m.fit(X_tr[cols].iloc[tri], y_tr[tri])
        probs_base += m.predict_proba(feat_te[cols])[:, 1] / kf.n_splits
        
    print(f"[{lang}] Model Base Prediction done. Launching Appeal Pipeline (K-NN + Groq LLM)...")
    
    miner = CandidateMiner(proba_low=0.10, proba_high=0.45, forensic_percentile=60, use_veto=True)
    court = PairwiseCourt(min_bot_votes=2)
    
    p_df_all = pd.concat([test_ev["p"]] + [ev["p"] for ev in train_evs])
    
    preds_new, overrides = run_appeal_pipeline(
        uids=uids, probs_base=probs_base, feat_df=feat_te[cols], 
        forensic_df=foren_te, labels=pd.Series(0, index=uids), all_cols=cols,
        miner=miner, court=court, verbose=False, posts_df=p_df_all,
        train_feat=X_tr, train_labels=pd.Series(y_tr, index=X_tr.index)
    )
    
    tp = np.sum((preds_new == 1) & (y_true == 1))
    fp = np.sum((preds_new == 1) & (y_true == 0))
    fn = np.sum((preds_new == 0) & (y_true == 1))
    
    print(f"--- {lang} RESULTS ---")
    print(f"TP={tp}, FN={fn}, FP={fp}")
    
    # Tracer les overrides Groq
    llm_actions = {k: v for k, v in overrides.items() if "groq_" in str(v.get("override", ""))}
    if llm_actions:
        print("\n=> Groq Arbitrator Rescues / Vetoes :")
        u_df = test_ev["u"]
        for uid, info in llm_actions.items():
            runame = u_df[u_df["user_id"] == uid]["username"]
            uname = runame.values[0] if len(runame) > 0 else uid
            print(f"   @{uname} : {info['override']} | LLM Confidence: {info.get('conf', 'N/A')}")
    else:
        print("\n=> No Groq Arbitrator overrides actively triggered.")

if __name__ == "__main__":
    evaluate_fold("EN", 5, [1, 3, 30], EN_MONOLITH_CONFIG)
    evaluate_fold("FR", 6, [2, 4, 31], FR_MONOLITH_CONFIG)
