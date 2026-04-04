import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.champion_config import EN_MONOLITH_CONFIG, FR_MONOLITH_CONFIG, get_lgbm_params, get_kfold_params
from src.features.candidate_miner_court import PairwiseCourt
from src.features.groq_forced_judge import GroqForcedJudge
from scripts.residual_error_autopsy import load_event, extract_features

EVENTS_EN = [1, 3, 30, 5]
EVENTS_FR = [2, 4, 31, 6]

def run_sensitivity_audit():
    print("Initiating Residual-Only LLM Sensitivity Audit...")

    results_table = []
    
    for lang, test_ev_id, train_ev_ids, config in [("EN", 5, [1,3,30], EN_MONOLITH_CONFIG), ("FR", 6, [2,4,31], FR_MONOLITH_CONFIG)]:
        te_ev = load_event(test_ev_id)
        tr_evs = [load_event(x) for x in train_ev_ids]
        
        te_f = extract_features(te_ev, config)
        tr_fs = [extract_features(e, config) for e in tr_evs]
        
        X_tr = pd.concat([t["feat"] for t in tr_fs])
        y_tr = np.concatenate([t["labels"].values for t in tr_fs])
        
        cols = [c for c in X_tr.columns if c in te_f["feat"].columns and c not in ["is_bot", "account_id"]]
        
        kf_p = get_kfold_params()
        kf = StratifiedKFold(**kf_p)
        lgbm_p = get_lgbm_params()
        
        uids = list(te_f["feat"].index)
        y_true = te_f["labels"].loc[uids].values
        probs_base = np.zeros(len(uids))
        
        for tri, _ in kf.split(X_tr[cols], y_tr):
            m = LGBMClassifier(**lgbm_p)
            m.fit(X_tr[cols].iloc[tri], y_tr[tri])
            probs_base += m.predict_proba(te_f["feat"][cols])[:, 1] / kf.n_splits
            
        preds = (probs_base >= 0.5).astype(int)
        errors_idx = np.where(preds != y_true)[0]
        
        if len(errors_idx) == 0:
            continue
            
        court = PairwiseCourt(min_bot_votes=2)
        p_df_all = pd.concat([te_ev["p"]] + [e["p"] for e in tr_evs])
        
        forced_judge = GroqForcedJudge(u_df=te_ev["u"], p_df=p_df_all, train_feat=X_tr, train_labels=pd.Series(y_tr, index=X_tr.index))
        
        for idx in errors_idx:
            uid = uids[idx]
            prob = probs_base[idx]
            is_fp = (preds[idx] == 1 and y_true[idx] == 0)
            is_fn = (preds[idx] == 0 and y_true[idx] == 1)
            err_type = "FP" if is_fp else "FN"
            
            c_res = court.adjudicate(uid, te_f["feat"].loc[[uid]], te_f["labels"], cols)
            bot_votes = c_res["bot_votes"]
            hum_votes = c_res.get("human_votes", 0)
            
            user_row = te_ev["u"][te_ev["u"]["user_id"] == uid]
            uname = user_row["username"].values[0] if not user_row.empty else uid
            
            # Appeler Groq
            if is_fp:
                res = forced_judge.evaluate_fp_veto(uid, prob, c_res["spaces"], te_f["feat"].loc[[uid]])
                v = res.get("verdict", "ERROR")
                c = res.get("confidence", 0.0)
                m = res.get("margin", 0.0)
                # Utile si = HUMAN, Dangereux si = BOT
                util = "UTIL" if v == "HUMAN" else "--"
                
                req_h = hum_votes if (v == "HUMAN" and hum_votes > 0) else "N/A"
                req_c = round(c - 0.01, 2) if v == "HUMAN" else "N/A"
                req_m = round(m - 0.01, 2) if v == "HUMAN" else "N/A"
                
            else:
                res = forced_judge.evaluate_fn_rescue(uid, prob, c_res["spaces"], te_f["feat"].loc[[uid]])
                v = res.get("verdict", "ERROR")
                c = res.get("confidence", 0.0)
                m = res.get("margin", 0.0)
                util = "UTIL" if v == "BOT" else "--"
                
                req_h = bot_votes if (v == "BOT" and bot_votes > 0) else "N/A"
                req_c = round(c - 0.01, 2) if v == "BOT" else "N/A"
                req_m = round(m - 0.01, 2) if v == "BOT" else "N/A"
                
            results_table.append({
                "Lang": lang, "Account": f"@{uname}", "Type": err_type,
                "Votes(Bot/Hum)": f"{bot_votes}/{hum_votes}",
                "Verdict": v, "Conf": c, "Margin": m,
                "Min Req": f"votes>={req_h}, conf>={req_c}, marg>={req_m}",
                "Status": util
            })

    print("\n\n" + "="*80)
    print("🤖 SENSITIVITY AUDIT TABLE".center(80))
    print("="*80)
    print(f"{'Lang':<5} | {'Account':<20} | {'Type':<4} | {'Votes(B/H)':<12} | {'LLM Verdict':<12} | {'Conf':<5} | {'Margin':<6} | {'Min Required Thresholds (if USEFUL)':<35} | {'Status'}")
    print("-" * 130)
    
    util_count = 0
    for r in results_table:
        if r['Status'] == "UTIL": util_count += 1
        print(f"{r['Lang']:<5} | {r['Account']:<20} | {r['Type']:<4} | {r['Votes(Bot/Hum)']:<12} | {r['Verdict']:<12} | {r['Conf']:<5} | {r['Margin']:<6} | {r['Min Req']:<35} | {r['Status']}")
        
    print("\n" + "="*80)
    print(f"Total utiles (Potentiel de corriger des résiduels) : {util_count}")
    if util_count > 0:
        print("\nVERDICT GLOBAL: »»» tiny_potential_found «««")
        print("Il existe une micro-zone où le LLM converge utilement avec la Cour.")
    else:
        print("\nVERDICT GLOBAL: »»» archive_definitive «««")
        print("Même en mode forcé, le LLM ne parie pas sur les résiduels, ou la Cour ne lui donne pas d'ancrage.")

if __name__ == "__main__":
    run_sensitivity_audit()
