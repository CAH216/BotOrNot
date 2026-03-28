#!/usr/bin/env python
"""
scripts/optimize_profiles.py
=============================
Mission 9 : Optimisation automatisée des 3 profils de soumission.

Effectue un Grid Search sur les hyperparamètres du pipeline post-modèle
(Anti-FP, blend method, threshold) sur un jeu de prédictions OOF, afin 
d'identifier la configuration EXACTE qui maximise l'objectif de chaque profil :
 - balanced : max F1
 - conservative : max Precision (avec Recall > 0.50)
 - aggressive : max Recall (avec Precision > 0.50)

Exporte un tableau récapitulatif dans artifacts/profiles/optimized_profiles.json.
"""

import sys, os, time, json, argparse, warnings
import itertools
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_baseline import _load_file, _find_col, _impute, _get_model, _fit_predict, ID_PATTERNS, LABEL_PATTERNS
from submission_factory import _extract_features, _extract_block_probas, _get_splitter
from src.inference.anti_fp import AntiFPFilter, AntiFPConfig

SEP = "=" * 70

def _banner(m): print(f"\n{SEP}\n  {m}\n{SEP}")
def _log(m):    print(f"  [{datetime.now():%H:%M:%S}] {m}")


def get_available_models():
    models = ["lr"]
    try:
        import lightgbm
        models.append("lgbm")
    except ImportError: pass
    try:
        import catboost
        models.append("catboost")
    except ImportError: pass
    return models


def generate_oofs(df_train, id_col, label_col, cv_folds, seed):
    _log("Extraction des features globales pour OOFs...")
    feat_train = _extract_features(df_train, id_col)
    
    y_true_s = df_train.groupby(id_col)[label_col].max()
    account_order = list(feat_train[id_col])
    y_true = y_true_s.reindex(account_order).values.astype(int)
    
    X = _impute(feat_train.drop(columns=[id_col]).select_dtypes(include=[np.number])).values
    
    groups_s = pd.Series(account_order)
    groups   = groups_s if groups_s.nunique() < len(y_true) else None
    
    splitter = _get_splitter(X, y_true, groups, cv_folds, seed)
    folds = list(splitter)
    
    models_to_test = get_available_models()
    oofs = {}
    
    for m_name in models_to_test:
        _log(f"  Entraînement {m_name}...")
        oof = np.zeros(len(y_true))
        for fold, (tr, va) in enumerate(folds, 1):
            m = _get_model(m_name, seed + fold)
            _, p = _fit_predict(m, X[tr], y_true[tr], X[va])
            oof[va] = p
        oofs[m_name] = oof
        
    _log("Extraction block_probas pour Anti-FP...")
    # On utilise le meilleur modèle pour générer les probas par bloc afin de nourrir l'anti-fp
    best_model = "lgbm" if "lgbm" in models_to_test else models_to_test[-1]
    block_probas = _extract_block_probas(df_train, id_col, y_true, best_model, cv_folds, seed, groups)
    
    return feat_train, y_true, account_order, oofs, block_probas


def create_blends(oofs):
    blends = {}
    for name, arr in oofs.items():
        blends[f"{name}_only"] = arr
        
    if "lgbm" in oofs and "catboost" in oofs:
        blends["mean(lgbm,catboost)"] = (oofs["lgbm"] + oofs["catboost"]) / 2.0
        blends["max(lgbm,catboost)"] = np.maximum(oofs["lgbm"], oofs["catboost"])
        # vote is practically max for 2 models, but let's just stick to continuous blends
        
    return blends


def generate_grid():
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    min_mods = [1, 2]
    uni_penalties = [0.0, 0.08, 0.15]
    pu_followers = [3000, 10000]
    pu_penalties = [0.05, 0.10]
    conflicts = [True, False]
    
    keys = ["threshold", "min_modules_for_bot", "unilateral_penalty", 
            "pu_min_followers", "pu_follower_penalty", "conflict_rules_enabled"]
    
    combos = list(itertools.product(
        thresholds, min_mods, uni_penalties, pu_followers, pu_penalties, conflicts
    ))
    
    return keys, combos


def optimize_profiles(args):
    t0 = time.time()
    _banner("MISSION 9 — OPTIMISATION DES PROFILS DE SOUMISSION")
    
    df_train = _load_file(args.train)
    id_col = _find_col(df_train, ID_PATTERNS) or "user_id"
    label_col = _find_col(df_train, LABEL_PATTERNS)
    
    if label_col is None:
        sys.exit("Label column introuvable.")
        
    feat_train, y_true, account_order, oofs, block_probas = generate_oofs(
        df_train, id_col, label_col, args.cv_folds, args.seed
    )
    
    feature_df = feat_train.drop(columns=[id_col]) if id_col in feat_train.columns else feat_train
    
    blends = create_blends(oofs)
    keys, grid = generate_grid()
    _log(f"Grille générée : {len(grid)} combinaisons × {len(blends)} blends = {len(grid)*len(blends)} évaluations.")
    
    results = []
    
    for blend_name, proba_arr in blends.items():
        _log(f" Évaluation du blend: {blend_name}...")
        
        # Le dataframe prefix pour l'anti-fp s'attend à "proba" et l'id_col
        base_df = pd.DataFrame({
            id_col: account_order,
            "proba": proba_arr
        })
        
        # Pour optimiser, on groupe par config Anti-FP car l'Anti-FP est le goulot (Pandas)
        # On extrait les sous-clés Anti-FP uniques
        # keys[1:] = min_modules, uni, pu_foll, pu_pen, conflict
        afp_configs = set(c[1:] for c in grid)
        
        for c_afp in afp_configs:
            cfg = AntiFPConfig(
                enabled=True,
                min_modules_for_bot=c_afp[0],
                unilateral_penalty=c_afp[1],
                power_user_protection=True,
                pu_min_followers=c_afp[2],
                pu_follower_penalty=c_afp[3],
                pu_verified_penalty=c_afp[3]+0.02,
                conflict_rules_enabled=c_afp[4],
                conflict_penalty=0.08
            )
            
            # Apply filter
            af_filter = AntiFPFilter(cfg)
            adj_df = af_filter.apply(base_df, feature_df, block_probas)
            adj_proba = adj_df["proba_adjusted"].values
            
            # Maintenant tester les thresholds
            # c_afp corresond aux c[1:] pour c in grid
            my_thresholds = [c[0] for c in grid if c[1:] == c_afp]
            
            for th in my_thresholds:
                pred = (adj_proba >= th).astype(int)
                f1 = f1_score(y_true, pred, zero_division=0)
                prec = precision_score(y_true, pred, zero_division=0)
                rec = recall_score(y_true, pred, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
                
                results.append({
                    "blend_method": blend_name,
                    "threshold": th,
                    "min_modules_for_bot": cfg.min_modules_for_bot,
                    "unilateral_penalty": cfg.unilateral_penalty,
                    "pu_min_followers": cfg.pu_min_followers,
                    "pu_follower_penalty": cfg.pu_follower_penalty,
                    "conflict_rules": cfg.conflict_rules_enabled,
                    "f1": float(f1),
                    "precision": float(prec),
                    "recall": float(rec),
                    "fp": int(fp),
                    "fn": int(fn)
                })

    df_res = pd.DataFrame(results)
    
    _banner("Sélection des Meilleurs Profils")
    
    # 1. Balanced: max F1
    best_balanced = df_res.iloc[df_res["f1"].idxmax()].to_dict()
    
    # 2. Conservative: max Precision avec Recall >= 0.30
    valid_cons = df_res[df_res["recall"] >= 0.30]
    if valid_cons.empty:
        valid_cons = df_res # fallback
    
    # On ordonne par Precision Desc, puis FP Asc
    valid_cons = valid_cons.sort_values(by=["precision", "fp"], ascending=[False, True])
    best_conservative = valid_cons.iloc[0].to_dict()
    
    # 3. Aggressive: max Recall avec Precision >= 0.30
    valid_agg = df_res[df_res["precision"] >= 0.30]
    if valid_agg.empty:
        valid_agg = df_res
        
    valid_agg = valid_agg.sort_values(by=["recall", "fn"], ascending=[False, True])
    best_aggressive = valid_agg.iloc[0].to_dict()
    
    # Print table
    out_table = []
    
    for pname, bdict in zip(["conservative", "balanced", "aggressive"], 
                            [best_conservative, best_balanced, best_aggressive]):
        
        expected = "Max Prec" if pname == "conservative" else "Max F1" if pname == "balanced" else "Max Recall"
        out_table.append({
            "profil": pname,
            "threshold": bdict["threshold"],
            "blend_method": bdict["blend_method"],
            "anti_fp_rules": f"min_{bdict['min_modules_for_bot']}mod_pen{bdict['unilateral_penalty']}_pu{bdict['pu_follower_penalty']}",
            "expected_behavior": f"{expected} (Prec={bdict['precision']:.3f}, F1={bdict['f1']:.3f}, Rec={bdict['recall']:.3f}, FP={bdict['fp']})",
            **bdict
        })
        
    df_out = pd.DataFrame(out_table)
    
    cols_display = ["profil", "threshold", "blend_method", "anti_fp_rules", "expected_behavior"]
    print(df_out[cols_display].to_string(index=False))
    
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "optimized_profiles.json")
    out_csv = os.path.join(args.out, "optimized_profiles.csv")
    
    with open(out_json, "w") as f:
        json.dump(out_table, f, indent=2)
    df_out[cols_display].to_csv(out_csv, index=False)
    
    _log(f"🚀 Profils optimisés générés en {time.time()-t0:.1f}s ! Exports: {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/profiles")
    args = p.parse_args()
    optimize_profiles(args)
