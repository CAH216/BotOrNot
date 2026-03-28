#!/usr/bin/env python
"""
scripts/coordination_v15_benchmark.py
======================================
Benchmark de la V1.0 contre la V1.5 avec Coordination (Rules.md).

Compare en OOF :
  - Pipeline V1.0 courant
  - Pipeline V1.5 (ajout module Coordination)

Exporte les résultats de l'ablation.
"""
import sys, os, time, json, argparse, warnings
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
    _load_file, _find_col, _impute, _get_model, _fit_predict, _build_features,
    ID_PATTERNS, LABEL_PATTERNS,
)
from src.preprocessing.normalize_columns import normalize_columns
from src.features.assembler import FeatureAssembler
from src.features.tabular import extract_tabular_features
from src.features.temporal import extract_temporal_features
from src.features.text_basic import extract_text_features
from src.features.structural import extract_structural_features
from src.features.coordination import extract_coordination_features


def _best_threshold(y, proba):
    best_t, best_f1 = 0.50, 0.0
    for t in np.arange(0.25, 0.80, 0.02):
        f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def _metrics(y, proba, label="") -> dict:
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


def _run_experiment(df_raw, id_col, label_col, use_coord, seeds, n_folds):
    feat_v1 = _build_features(df_raw.copy(), id_col)
    
    if use_coord:
        # The raw dataframe has created_at and account_id from the CSV, which match our schema patterns usually
        coords = extract_coordination_features(df_raw, df_raw, {"coordination": {"enabled": True}})
        if coords is not None and not coords.empty:
            # Rename account_id if needed to match id_col in feat_v1
            if "account_id" in coords.columns and id_col != "account_id":
                coords = coords.rename(columns={"account_id": id_col})
            feat_v1 = feat_v1.merge(coords, on=id_col, how="left")

    ids = list(feat_v1[id_col])
    y = df_raw.groupby(id_col)[label_col].max().reindex(ids).values.astype(int)
    X = _impute(feat_v1.drop(columns=[id_col], errors="ignore").select_dtypes(include=[np.number])).values
    
    groups = pd.Series(ids)
    all_metrics = []
    
    for seed in seeds:
        if len(np.unique(groups)) >= n_folds * 2:
            splitter = list(GroupKFold(n_splits=n_folds).split(X, y, groups=groups))
        else:
            splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y))
            
        oof = np.zeros(len(y))
        for fold, (tr, va) in enumerate(splitter, 1):
            m = _get_model("lr", seed + fold)
            _, p = _fit_predict(m, X[tr], y[tr], X[va])
            oof[va] = p
        
        all_metrics.append(_metrics(y, oof))
        
    # Aggregate
    out = {}
    for m in all_metrics[0].keys():
        arr = [res[m] for res in all_metrics]
        out[m] = {"mean": np.mean(arr), "std": np.std(arr)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--out", default="artifacts/coordination_v15")
    args = p.parse_args()

    df = _load_file(args.train)
    id_col = _find_col(df, ID_PATTERNS) or "user_id"
    label_col = _find_col(df, LABEL_PATTERNS)
    
    seeds = [42 + i*17 for i in range(args.n_seeds)]
    
    print(f"Running Baseline...")
    base_res = _run_experiment(df, id_col, label_col, False, seeds, args.cv_folds)
    print(f"Running V1.5 (with Coordination)...")
    coord_res = _run_experiment(df, id_col, label_col, True, seeds, args.cv_folds)
    
    os.makedirs(args.out, exist_ok=True)
    
    delta_auroc = coord_res["auroc"]["mean"] - base_res["auroc"]["mean"]
    sigma = coord_res["auroc"]["std"]
    delta_fp = coord_res["fp"]["mean"] - base_res["fp"]["mean"]
    
    decision = "reject"
    if sigma <= 0.02 and delta_auroc >= 0.01 and delta_fp <= 0:
        decision = "activate"
        
    rep = {
        "V1_baseline": base_res,
        "V1_5_Coordination": coord_res,
        "delta_AUROC": delta_auroc,
        "delta_FP": delta_fp,
        "sigma": sigma,
        "DECISION": decision
    }
    
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
        
    print(f"Decision: {decision}")
    print(f"Delta AUROC: {delta_auroc:+.4f}")
    if decision == "activate":
        print("RECOMMENDATION: Replace features.yaml coordination node to enabled=True")

if __name__ == "__main__":
    main()
