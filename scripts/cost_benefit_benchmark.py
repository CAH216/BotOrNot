#!/usr/bin/env python
"""
scripts/cost_benefit_benchmark.py
==================================
Benchmark de la Mission 8: Coût/Bénéfice par module (LOO-FI)

Mesure l'impact marginal de chaque module sur les performances du modèle
en comparant le pipeline complet ("Full") contre le pipeline (Full - Module).
Mesure également le coût CPU et RAM pur d'extraction pour statuer sur l'activation.

S'exécute sur le dataset complet. Produit un rapport JSON et CSV.
"""

import sys, os, time, json, argparse, warnings
import tracemalloc
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_baseline import (
    _load_file, _find_col, _impute, _get_model, _fit_predict,
    ID_PATTERNS, LABEL_PATTERNS,
)
from src.preprocessing.normalize_columns import normalize_columns

# Extracts imports
from src.features.assembler import FeatureAssembler
from src.features.tabular import extract_tabular_features
from src.features.temporal import extract_temporal_features
from src.features.text_basic import extract_text_features
from src.features.text_embeddings import extract_text_embeddings
from src.features.structural import extract_structural_features
from src.features.relational import extract_relational_features
from src.features.coordination import extract_coordination_features


MODULES_INFO = {
    "tabular":         {"deps": "pandas, numpy"},
    "temporal":        {"deps": "pandas, numpy"},
    "text_basic":      {"deps": "pandas, re, string"},
    "text_embeddings": {"deps": "sentence-transformers, torch, pandas"},
    "structural":      {"deps": "networkx, pandas, numpy"},
    "relational":      {"deps": "networkx, pandas, numpy"},
    "coordination":    {"deps": "pandas, numpy"},
}

def _best_threshold(y, proba):
    best_t, best_f1 = 0.50, 0.0
    for t in np.arange(0.25, 0.80, 0.02):
        f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def _metrics(y, proba) -> dict:
    if len(np.unique(y)) < 2:
        return {m: 0.0 for m in ["auroc","pr_auc","f1","precision","recall","fp","fn"]}
    t    = _best_threshold(y, proba)
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc":     float(roc_auc_score(y, proba)),
        "pr_auc":    float(average_precision_score(y, proba)),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "fp":        int(fp),
        "fn":        int(fn),
    }


def cv_evaluate(X, y, groups, n_folds=3, seed=42):
    if groups is not None and len(np.unique(groups)) >= n_folds * 2:
        splitter = list(GroupKFold(n_splits=n_folds).split(X, y, groups=groups))
    else:
        splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y))
        
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splitter, 1):
        m = _get_model("lr", seed + fold)
        _, p = _fit_predict(m, X[tr], y[tr], X[va])
        oof[va] = p
        
    return _metrics(y, oof)


def measure_memory_mb(df):
    if df is None or df.empty:
        return 0.0
    mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    return round(mem, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/cost_benefit")
    args = p.parse_args()

    print(f"\n[+] Loading data from {args.train}...")
    df_raw = _load_file(args.train)
    id_col = _find_col(df_raw, ID_PATTERNS) or "user_id"
    label_col = _find_col(df_raw, LABEL_PATTERNS)
    
    df, _ = normalize_columns(df_raw.copy())
    if id_col not in df.columns and "account_id" in df.columns:
        norm_id_col = "account_id"
    else:
        norm_id_col = id_col

    # Dictionary to hold the feature dataframes
    extracted_dfs = {}
    costs = {}

    extractors = {
        "tabular": lambda: extract_tabular_features(df),
        "temporal": lambda: extract_temporal_features(df),
        "text_basic": lambda: extract_text_features(df),
        "text_embeddings": lambda: extract_text_embeddings(df),
        "structural": lambda: extract_structural_features(df, None, {"structural": {
            "source_v11_enabled": True, "batch_v11_enabled": True,
            "profile_v11_enabled": True, "template_v11_enabled": True}}),
        "relational": lambda: extract_relational_features(df, None, df, None),
        "coordination": lambda: extract_coordination_features(df, df, {"coordination": {"enabled": True}}),
    }

    print("[+] Phase 1: Resource Cost Measurement (CPU Time & RAM)")
    for mod_name, ext_func in extractors.items():
        tracemalloc.start()
        t0 = time.time()
        
        try:
            res_df = ext_func()
            t1 = time.time()
            mem_mb = measure_memory_mb(res_df)
            
            extracted_dfs[mod_name] = res_df
            costs[mod_name] = {
                "runtime_cost": round(t1 - t0, 3),
                "memory_mb": mem_mb,
                "n_cols": len(res_df.columns) - 1 if res_df is not None and not res_df.empty else 0,
                "deps": MODULES_INFO[mod_name]["deps"]
            }
            print(f"  -> {mod_name:<15}: {costs[mod_name]['runtime_cost']:>6.2f}s | {mem_mb:>5.2f} MB")
        except Exception as e:
            print(f"  -> {mod_name:<15}: FAILED ({e})")
            costs[mod_name] = {"runtime_cost": 0.0, "memory_mb": 0.0, "deps": MODULES_INFO[mod_name]["deps"], "error": str(e)}
        finally:
            tracemalloc.stop()

    print("\n[+] Phase 2: Assembling Full Dataset")
    asm = FeatureAssembler(account_id_col=norm_id_col)
    
    # Store the exact columns each module contributes to be able to ablate them correctly
    module_cols = {}
    
    for mod_name, res_df in extracted_dfs.items():
        if res_df is not None and not res_df.empty:
            asm.add_block(mod_name, res_df)
            module_cols[mod_name] = [c for c in res_df.columns if c != norm_id_col]

    X_full_df, _, _, _ = asm.assemble(return_labels=False)
    ids = list(X_full_df.index)
    groups = pd.Series(ids)
    
    # Reindex target
    y = df_raw.groupby(id_col)[label_col].max().reindex(ids).fillna(0).values.astype(int)
    
    # Impute full dataset
    X_full = _impute(X_full_df.select_dtypes(include=[np.number])).values
    
    print(f"  -> Full matrix shape: {X_full.shape}")

    print("\n[+] Phase 3: Ablation (Leave-One-Out Feature Importance)")
    print("  -> Evaluating Full Pipeline...")
    full_metrics = cv_evaluate(X_full, y, groups, n_folds=args.cv_folds, seed=args.seed)
    
    print(f"  -> FULL AUROC: {full_metrics['auroc']:.4f}")

    results_table = []

    for mod_name in extractors.keys():
        if mod_name not in module_cols or not module_cols[mod_name]:
            # Module failed or empty
            results_table.append({
                "module": mod_name,
                "gain_auroc": 0.0,
                "gain_precision": 0.0,
                "delta_fp": 0,
                "delta_fn": 0,
                "runtime_cost_s": costs.get(mod_name, {}).get("runtime_cost", 0.0),
                "memory_mb": costs.get(mod_name, {}).get("memory_mb", 0.0),
                "deps": costs.get(mod_name, {}).get("deps", ""),
                "keep_default": False
            })
            continue
            
        print(f"  -> Evaluating Pipeline WITHOUT '{mod_name}'...")
        # Drop columns belonging to this module
        cols_to_drop = [c for c in module_cols[mod_name] if c in X_full_df.columns]
        X_ablated_df = X_full_df.copy().drop(columns=cols_to_drop)
        X_abl_imputed = _impute(X_ablated_df.select_dtypes(include=[np.number])).values
        
        abl_metrics = cv_evaluate(X_abl_imputed, y, groups, n_folds=args.cv_folds, seed=args.seed)
        
        # Calculate impact
        # Gain is how much the FULL model drops when we remove the module M
        gain_auroc = full_metrics['auroc'] - abl_metrics['auroc']
        gain_precision = full_metrics['precision'] - abl_metrics['precision']
        
        # Delta FP / FN: if we remove the module, FP goes up -> module prevents FP
        # So "delta_fp" = abl_metrics['fp'] - full_metrics['fp'] ?
        # The prompt says delta_fp in the formula, but reduction_fp. Let's output reduction:
        # positive reduction_fp = LOO FP - FULL FP
        reduction_fp = abl_metrics['fp'] - full_metrics['fp']
        reduction_fn = abl_metrics['fn'] - full_metrics['fn']
        
        runtime = costs[mod_name]["runtime_cost"]
        
        # Heuristic for keep_default
        keep = False
        if runtime < 10.0 and gain_auroc > -0.005 and reduction_fp >= -10: 
            # Cheap and doesn't hurt much
            keep = True
        elif gain_auroc >= 0.01 or reduction_fp >= 10:
            # Expensive but highly valuable
            keep = True
            
        results_table.append({
            "module": mod_name,
            "gain_auroc": round(gain_auroc, 4),
            "gain_precision": round(gain_precision, 4),
            "delta_fp": reduction_fp,  # Positive means it prevents FP
            "delta_fn": reduction_fn,
            "runtime_cost_s": runtime,
            "memory_mb": costs[mod_name]["memory_mb"],
            "deps": costs[mod_name]["deps"],
            "keep_default": keep
        })
        
    print("\n[+] Phase 4: Final Report")
    df_report = pd.DataFrame(results_table)
    print(df_report.to_string(index=False))
    
    # Make dir and Export
    os.makedirs(args.out, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out, f"benchmark_{ts}.csv")
    json_path = os.path.join(args.out, f"benchmark_{ts}.json")
    
    df_report.to_csv(csv_path, index=False)
    
    # Save a detailed json including absolute full metrics
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.train,
        "full_metrics": full_metrics,
        "modules_benchmark": results_table
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\n[+] Exported to:\n  - {csv_path}\n  - {json_path}")
    
if __name__ == "__main__":
    main()
