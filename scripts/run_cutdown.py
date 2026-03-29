#!/usr/bin/env python
"""
run_cutdown.py — Baseline ultra-rapide (mode urgence)
======================================================
Pipeline minimaliste : tabular + temporal + LightGBM.
Aucune dépendance optionnelle. Prêt en 2–8 minutes.

Usage :
    python scripts/run_cutdown.py --train data/train.csv
    python scripts/run_cutdown.py --train data/train.csv --test data/test.csv
    python scripts/run_cutdown.py --train data/train.csv --test data/test.csv \\
        --profile balanced --cv-folds 3
"""
import sys, os, time, json, argparse, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from run_baseline import (
    _load_file, _find_col,
    _make_tabular_features, _make_temporal_features,
    _impute, _get_model, _fit_predict,
    ID_PATTERNS, LABEL_PATTERNS,
)

SEP = "─" * 68

# ── Profils intégrés (seuils + règle) ────────────────────────
CUTDOWN_PROFILES = {
    "conservative": {
        "threshold":  0.60,
        "description": "Précision max — seuil haut, moins de faux positifs",
    },
    "balanced": {
        "threshold":  None,          # auto F1-optimal sur OOF
        "description": "F1 optimal — seuil auto sur validation",
    },
}


def _banner(m): print(f"\n{SEP}\n  {m}\n{SEP}")
def _log(m):    print(f"  [{datetime.now():%H:%M:%S}] {m}")


def _best_threshold(y, proba):
    best_t, best_f1 = 0.50, 0.0
    for t in np.arange(0.25, 0.80, 0.02):
        f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    # Garde-fous
    return round(float(max(0.45, min(0.65, best_t))), 3)


def _metrics_summary(y, proba, t) -> dict:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc":     round(float(roc_auc_score(y, proba)), 4),
        "pr_auc":    round(float(average_precision_score(y, proba)), 4),
        "f1":        round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, pred, zero_division=0)), 4),
        "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
        "threshold": t,
    }


def run_cutdown(args):
    t_global = time.time()
    _banner(f"⚡  CUT-DOWN BASELINE — BotOrNot  [{datetime.now():%H:%M:%S}]")

    # ── Profil ───────────────────────────────────────────────
    pcfg = CUTDOWN_PROFILES.get(args.profile)
    if not pcfg:
        sys.exit(f"❌ Profil inconnu : {args.profile}. Choisir : {list(CUTDOWN_PROFILES)}")
    _log(f"Profil : {args.profile} — {pcfg['description']}")

    # ── Chargement ───────────────────────────────────────────
    _log(f"Train : {args.train}")
    df = _load_file(args.train)
    _log(f"  {len(df):,} lignes × {len(df.columns)} colonnes")

    has_test = args.test and os.path.exists(args.test)
    if has_test:
        df_test = _load_file(args.test)
        _log(f"Test  : {len(df_test):,} lignes")
    else:
        _log("⚠️  Pas de test — inférence sur train (aucune soumission test)")
        df_test = None

    id_col    = args.id_col    or _find_col(df, ID_PATTERNS) or "user_id"
    label_col = args.label_col or _find_col(df, LABEL_PATTERNS)
    if label_col is None:
        sys.exit("❌ Colonne label introuvable. Utilisez --label-col.")

    # ── Features : tabular + temporal SEULEMENT ───────────────
    _banner("Extraction features (tabular + temporal)")
    t0 = time.time()

    tab = _make_tabular_features(df, id_col).groupby(id_col).first().reset_index()
    tmp = _make_temporal_features(df, id_col)

    base_ids = list(tab[id_col])
    feat = tab.merge(tmp, on=id_col, how="left")
    X_df = feat.drop(columns=[id_col]).select_dtypes(include=[np.number])
    X    = _impute(X_df).values

    y_s = df.groupby(id_col)[label_col].max()
    y   = y_s.reindex(base_ids).values.astype(int)

    _log(f"  {len(y)} comptes | {X.shape[1]} features | bots: {y.mean():.1%} | {time.time()-t0:.1f}s")

    # ── Modèle ───────────────────────────────────────────────
    model_name = args.model
    try:
        import lightgbm; has_lgbm = True
    except ImportError:
        has_lgbm = False
    if model_name == "lgbm" and not has_lgbm:
        _log("  LightGBM absent → Logistic Regression")
        model_name = "lr"

    _log(f"Modèle : {model_name.upper()}  |  {args.cv_folds}-fold CV")

    # ── CV + OOF ─────────────────────────────────────────────
    _banner(f"Cross-validation — {args.cv_folds} folds")
    t0 = time.time()

    groups_s = pd.Series(base_ids)
    if groups_s.nunique() >= args.cv_folds * 2:
        splitter = list(GroupKFold(n_splits=args.cv_folds).split(X, y, groups=groups_s))
    else:
        splitter = list(StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                                        random_state=args.seed).split(X, y))

    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splitter, 1):
        m = _get_model(model_name, args.seed + fold)
        _, p = _fit_predict(m, X[tr], y[tr], X[va])
        oof[va] = p
        _log(f"  Fold {fold}/{args.cv_folds}  AUC={roc_auc_score(y[va], p):.4f}")

    _log(f"  OOF AUROC : {roc_auc_score(y, oof):.4f}  ({time.time()-t0:.1f}s)")

    # ── Seuil ────────────────────────────────────────────────
    if pcfg["threshold"] is None:
        threshold = _best_threshold(y, oof)
        _log(f"  Seuil F1-optimal : {threshold:.3f}")
    else:
        threshold = pcfg["threshold"]
        _log(f"  Seuil fixe ({args.profile}) : {threshold:.3f}")

    # Appliquer le flag --threshold si explicitement fourni
    if args.threshold is not None:
        threshold = args.threshold
        _log(f"  Seuil overridé par --threshold : {threshold:.3f}")

    # ── Métriques OOF ────────────────────────────────────────
    m_oof = _metrics_summary(y, oof, threshold)
    _log(f"  AUROC={m_oof['auroc']:.4f}  F1={m_oof['f1']:.4f}  "
         f"Prec={m_oof['precision']:.4f}  Rec={m_oof['recall']:.4f}  "
         f"FP={m_oof['fp']}  FN={m_oof['fn']}")

    # ── Entraînement final sur tout le train ──────────────────
    _banner("Entraînement final + export")
    t0 = time.time()
    final = _get_model(model_name, args.seed)
    final.fit(X, y)

    # ── Inférence test ────────────────────────────────────────
    test_accounts = []
    if df_test is not None:
        tab_test = _make_tabular_features(df_test, id_col).groupby(id_col).first().reset_index()
        tmp_test = _make_temporal_features(df_test, id_col)
        feat_test = tab_test.merge(tmp_test, on=id_col, how="left")
        X_test_df = feat_test.drop(columns=[id_col]).select_dtypes(include=[np.number])

        # Aligner colonnes
        for c in X_df.columns:
            if c not in X_test_df.columns:
                X_test_df[c] = np.nan
        X_test_df = X_test_df[X_df.columns]
        X_test = _impute(X_test_df).values

        test_proba = final.predict_proba(X_test)[:, 1]
        test_labels = (test_proba >= threshold).astype(int)
        test_accounts = list(feat_test[id_col])

        submission = pd.DataFrame({
            id_col:    test_accounts,
            "proba":   np.round(test_proba, 6),
            label_col: test_labels,
        })
    else:
        # Pseudo-soumission sur train
        train_proba    = final.predict_proba(X)[:, 1]
        train_labels   = (train_proba >= threshold).astype(int)
        test_accounts  = base_ids
        submission = pd.DataFrame({
            id_col:    base_ids,
            "proba":   np.round(train_proba, 6),
            label_col: train_labels,
        })

    # ── Écriture ──────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    prefix    = f"cutdown_{args.profile}"
    csv_path  = os.path.join(args.out, f"{prefix}.csv")
    meta_path = os.path.join(args.out, f"{prefix}_meta.json")

    submission.to_csv(csv_path, index=False)

    meta = {
        "generated_at":      datetime.now().isoformat(),
        "profile":           args.profile,
        "model":             model_name,
        "cv_folds":          args.cv_folds,
        "seed":              args.seed,
        "threshold":         threshold,
        "features_used":     ["tabular", "temporal"],
        "n_features":        int(X.shape[1]),
        "n_train_accounts":  len(y),
        "n_test_accounts":   len(test_accounts),
        "bot_rate_test":     round(float(submission[label_col].mean()), 4),
        "oof_metrics":       m_oof,
        "elapsed_seconds":   round(time.time() - t_global, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    total = time.time() - t_global
    _banner(f"✅  Terminé en {total:.1f}s")
    _log(f"CSV  → {csv_path}  ({submission[label_col].sum()} bots / {len(submission)} comptes)")
    _log(f"Meta → {meta_path}")
    return meta


def main():
    p = argparse.ArgumentParser(description="⚡ Cut-down baseline BotOrNot (tabular+temporal)")
    p.add_argument("--train",       required=True)
    p.add_argument("--test",        default=None)
    p.add_argument("--profile",     default="conservative",
                   choices=["conservative", "balanced"])
    p.add_argument("--model",       default="lgbm", choices=["lgbm", "lr"])
    p.add_argument("--cv-folds",    type=int,   default=3)   # 3 par défaut pour la vitesse
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--threshold",   type=float, default=None,
                   help="Override le seuil du profil (ex: 0.55)")
    p.add_argument("--out",         default="artifacts/submissions")
    p.add_argument("--label-col",   default=None)
    p.add_argument("--id-col",      default=None)
    args = p.parse_args()
    run_cutdown(args)


if __name__ == "__main__":
    main()
