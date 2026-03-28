#!/usr/bin/env python
"""
scripts/premium_benchmark.py
============================
Mission 12 : Évaluation du Bloc "Premium mais Safe"

Vise à comparer statistiquement 2 pipelines via Validation Croisée :
 1. La "Golden Baseline" (features classiques).
 2. Le "Premium Candidate" (Golden + Coordination + Structural avancé).

Vérifie les contraintes de stabilité de RULES.md :
 - Si le Premium gagne en moyenne sans exploser les Faux Positifs : Victoire.
 - Dans ce cas, génère le fichier `configs/premium_candidate.yaml`.
"""

import sys, os, time, json, argparse, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from run_baseline import _load_file, _find_col, _impute, _get_model, _fit_predict, ID_PATTERNS, LABEL_PATTERNS, _make_tabular_features, _make_temporal_features, _make_text_features
from submission_factory import _get_splitter
from src.features.structural import extract_structural_features
from src.features.coordination import extract_coordination_features

SEP = "=" * 70

def _banner(m): print(f"\n{SEP}\n  {m}\n{SEP}")
def _log(m):    print(f"  [{datetime.now():%H:%M:%S}] {m}")


def get_feature_matrix(df, id_col, mode="baseline"):
    """
    mode='baseline' : tabular + temporal + text + structurel basique (sans v11)
    mode='premium'  : baseline + structurel avancé + coordination
    """
    tab = _make_tabular_features(df, id_col).groupby(id_col).first().reset_index()
    tmp = _make_temporal_features(df, id_col)
    txt = _make_text_features(df, id_col)
    
    # Structurel basique vs avancé
    struct_cfg = {
        "structural": {
            "source_v11_enabled": (mode == "premium"),
            "batch_v11_enabled": (mode == "premium"),
            "profile_v11_enabled": (mode == "premium"),
            "template_v11_enabled": (mode == "premium")
        }
    }
    struc = extract_structural_features(df, posts_df=df, cfg=struct_cfg)
    
    base = tab[[id_col]].copy()
    def _m(block):
        if block is None or block.empty:
            return pd.DataFrame(index=base.index)
        m = base.merge(block, on=id_col, how="left")
        return m.drop(columns=[id_col]).select_dtypes(include=[np.number])
        
    blocks = [_m(tab), _m(tmp), _m(txt), _m(struc)]
    
    # Coordination pour le premium
    if mode == "premium":
        coord_cfg = {
            "coordination": {
                "enabled": True,
                "time_window_minutes": 60,
                "min_users_per_bin": 2
            }
        }
        coord = extract_coordination_features(df, posts_df=df, cfg=coord_cfg)
        blocks.append(_m(coord))
        
    t = pd.concat(blocks, axis=1)
    t = t.loc[:, ~t.columns.duplicated()]
    return t


def compare_pipelines(args):
    t0 = time.time()
    _banner("MISSION 12 — BENCHMARK DU BLOC PREMIUM")
    
    df_train = _load_file(args.train)
    id_col = _find_col(df_train, ID_PATTERNS) or "user_id"
    label_col = _find_col(df_train, LABEL_PATTERNS)
    
    y_true_s = df_train.groupby(id_col)[label_col].max()
    account_order = list(y_true_s.index)
    y_true = y_true_s.values.astype(int)
    groups = pd.Series(account_order) if len(account_order) < len(y_true) else None
    
    n_seeds = args.seeds
    n_folds = args.cv_folds
    model_name = "lgbm"
    seeds = [42 + i for i in range(n_seeds)]
    
    _log(f"Dataset : {len(y_true)} comptes. Bots : {y_true.mean():.1%}")
    _log(f"Protocole : {n_seeds} Seeds × {n_folds} Folds = {n_seeds*n_folds} runs par variante.")
    
    results = {}
    
    for mode in ["baseline", "premium"]:
        _banner(f"ÉVALUATION DE LA VARIANTE : {mode.upper()}")
        t1 = time.time()
        
        feat_df = get_feature_matrix(df_train, id_col, mode=mode)
        X = _impute(feat_df).values
        n_feat = X.shape[1]
        _log(f" Matrice extraite : {n_feat} features ({time.time()-t1:.1f}s)")
        
        all_auroc = []
        all_fp = []
        
        for seed in seeds:
            oof = np.zeros(len(y_true))
            splitter = _get_splitter(X, y_true, groups, n_folds, seed)
            for fold, (tr, va) in enumerate(splitter, 1):
                m = _get_model(model_name, seed + fold)
                _, p = _fit_predict(m, X[tr], y_true[tr], X[va])
                oof[va] = p
                
            auc = roc_auc_score(y_true, oof)
            pred = (oof >= 0.50).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            
            all_auroc.append(auc)
            all_fp.append(fp)
            _log(f"   Seed {seed} | AUROC {auc:.4f} | FP {fp}")
            
        results[mode] = {
            "auroc_mean": np.mean(all_auroc),
            "auroc_std": np.std(all_auroc),
            "fp_mean": np.mean(all_fp)
        }
        
    _banner("RÉSULTAT DU MATCH")
    res_b = results["baseline"]
    res_p = results["premium"]
    
    auroc_gain = res_p["auroc_mean"] - res_b["auroc_mean"]
    fp_delta = res_p["fp_mean"] - res_b["fp_mean"]
    
    print(f"  Baseline : AUROC {res_b['auroc_mean']:.4f} ±{res_b['auroc_std']:.4f} | FP {res_b['fp_mean']:.1f}")
    print(f"  Premium  : AUROC {res_p['auroc_mean']:.4f} ±{res_p['auroc_std']:.4f} | FP {res_p['fp_mean']:.1f}")
    print(f"  Gain     : Delta AUROC {auroc_gain:+.4f} | Delta FP {fp_delta:+.1f}")
    
    # Règles de Validation RULES.md
    validated = False
    if auroc_gain > 0.005 and fp_delta <= (0.05 * res_b["fp_mean"]):
        validated = True
        reason = "Gain significatif de performance sans augmentation majeure des FP."
    elif auroc_gain > 0.0 and fp_delta < 0:
        validated = True
        reason = "Amélioration légère mais réduction nette des Faux Positifs."
    else:
        reason = "Gain insuffisant ou instable (Variance ou Faux Positifs en hausse)."
        
    if validated:
        print(f"\n✅ LE PREMIUM EST VALIDÉ ! Raison : {reason}")
        
        yaml_content = f"""# Configurations validées du PREMIUM CANDIDATE
# Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# Performance Benchmark:
#   Gain AUROC : +{auroc_gain:.4f}
#   Delta FP   : {fp_delta:+.1f}

features:
  coordination:
    enabled: true
    time_window_minutes: 60
    min_users_per_bin: 2

  structural:
    enabled: true
    components:
      source_analysis: true
      batch_posting: true
      profile_consistency: true
      template_detection: true

inference:
  consensus_v2:
    enabled: true
    bot_threshold: 0.50
    agreement_weight: 0.25
    confidence_weight: 0.25
    spread_penalty_weight: 0.20
    v2_alpha: 0.50
"""
        yaml_path = os.path.join("configs", "premium_candidate.yaml")
        os.makedirs("configs", exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        _log(f"Fichier exporté avec succès : {yaml_path}")
    else:
        print(f"\n❌ LE PREMIUM EST REJETÉ ! Raison : {reason}")
        print("Maintien strict de la Golden Baseline en production.")

    _log(f"Exécution totale : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    compare_pipelines(args)
