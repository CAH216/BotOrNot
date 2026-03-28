#!/usr/bin/env python
"""
experiment_runner.py — Harness d'expérimentation contrôlée
===========================================================
Compare un pipeline baseline (golden_baseline.yaml) vs un pipeline
candidat (experiment YAML) sur les mêmes données et folds.

Garanties (RULES.md) :
  - Ne modifie JAMAIS golden_baseline.yaml
  - Le candidat est un overlay partiel sur la baseline
  - Même seed, mêmes folds pour les deux runs
  - Décision automatique : keep / reject / investigate

Usage :
    # Créer d'abord un experiment YAML (overlay partiel)
    python scripts/experiment_runner.py \\
        --train data/train.csv \\
        --experiment configs/experiments/xgb_model.yaml

    # Avec comparaison sur données de test
    python scripts/experiment_runner.py \\
        --train data/train.csv \\
        --test  data/test.csv  \\
        --experiment configs/experiments/text_model_enabled.yaml \\
        --baseline configs/golden_baseline.yaml \\
        --out artifacts/experiments

Format d'un fichier experiment YAML :
    # configs/experiments/my_experiment.yaml
    meta:
      name: "test_xgb_model"
      description: "Remplacer lgbm par xgboost"
      hypothesis: "XGBoost pourrait converger mieux sur ce dataset"

    # Seulement les champs qui diffèrent de la baseline
    model:
      name: "xgboost"
      hyperparameters:
        n_estimators: 500
        learning_rate: 0.05
        max_depth: 6
"""
import sys, os, time, json, argparse, copy, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
from run_baseline import (
    _load_file, _find_col,
    _make_tabular_features, _make_text_features, _make_temporal_features,
    _impute, _get_model, _fit_predict,
    ID_PATTERNS, LABEL_PATTERNS,
)

SEP  = "=" * 72
SEP2 = "-" * 72


def _log(msg): print(f"  [{datetime.now():%H:%M:%S}] {msg}")
def _banner(msg): print(f"\n{SEP}\n  {msg}\n{SEP}")
def _section(msg): print(f"\n{SEP2}\n  {msg}\n{SEP2}")


# ─────────────────────────────────────────────────────────────
# Config management (golden baseline + overlay)
# ─────────────────────────────────────────────────────────────

GOLDEN_BASELINE_PATH = "configs/golden_baseline.yaml"


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Fusionne overlay dans base (récursif). Ne modifie pas base."""
    result = copy.deepcopy(base)
    for k, v in overlay.items():
        if k == "meta":          # meta est ignoré pour le merge (info only)
            continue
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_experiment_config(baseline_path: str, experiment_path: str) -> tuple:
    """
    Retourne (baseline_cfg, candidate_cfg).
    candidate = baseline + overlay experiment.
    Ne touche jamais au fichier baseline.
    """
    baseline_cfg   = _load_yaml(baseline_path)
    experiment_cfg = _load_yaml(experiment_path)
    candidate_cfg  = _deep_merge(baseline_cfg, experiment_cfg)
    return baseline_cfg, candidate_cfg, experiment_cfg


def _diff_configs(base: dict, candidate: dict, prefix="") -> list:
    """Liste les clés qui diffèrent entre baseline et candidat."""
    diffs = []
    all_keys = set(base.keys()) | set(candidate.keys())
    for k in sorted(all_keys):
        full_key = f"{prefix}.{k}" if prefix else k
        bv, cv   = base.get(k), candidate.get(k)
        if isinstance(bv, dict) and isinstance(cv, dict):
            diffs.extend(_diff_configs(bv, cv, full_key))
        elif bv != cv:
            diffs.append({"key": full_key, "baseline": bv, "candidate": cv})
    return diffs


# ─────────────────────────────────────────────────────────────
# Feature extraction selon config
# ─────────────────────────────────────────────────────────────

def _extract_for_config(df: pd.DataFrame, id_col: str, cfg: dict) -> pd.DataFrame:
    feats = cfg.get("features", {})
    base  = _make_tabular_features(df, id_col).groupby(id_col).first().reset_index()
    frames = [base]

    if feats.get("temporal", True):
        tmp = _make_temporal_features(df, id_col)
        if not tmp.empty:
            frames.append(tmp)

    if feats.get("text_basic", True):
        txt = _make_text_features(df, id_col)
        if not txt.empty:
            frames.append(txt)

    merged = frames[0][[id_col]].copy()
    for fr in frames:
        merged = merged.merge(fr, on=id_col, how="left")

    X = merged.drop(columns=[id_col]).select_dtypes(include=[np.number])
    X = X.loc[:, X.isna().mean() < 0.95]  # drop near-empty cols
    X[id_col] = merged[id_col].values
    return X


# ─────────────────────────────────────────────────────────────
# Modèle depuis config
# ─────────────────────────────────────────────────────────────

def _get_model_from_config(cfg: dict, seed: int):
    model_cfg  = cfg.get("model", {})
    model_name = model_cfg.get("name", "lgbm")
    hyperparams = model_cfg.get("hyperparameters", {})

    # Résolution des dépendances
    try:
        import lightgbm; has_lgbm = True
    except ImportError:
        has_lgbm = False
    try:
        import catboost; has_cb = True
    except ImportError:
        has_cb = False
    try:
        import xgboost; has_xgb = True
    except ImportError:
        has_xgb = False

    if model_name == "lgbm" and has_lgbm:
        import lightgbm as lgb
        params = dict(hyperparams)
        params.setdefault("n_estimators", 300)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 31)
        params.setdefault("class_weight", "balanced")
        params.setdefault("verbose", -1)
        params.setdefault("n_jobs", -1)
        params["random_state"] = seed
        return lgb.LGBMClassifier(**params)

    if model_name in ("xgboost", "xgb") and has_xgb:
        import xgboost as xgb
        params = dict(hyperparams)
        params.setdefault("n_estimators", 300)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("max_depth", 6)
        params.setdefault("eval_metric", "logloss")
        params.setdefault("use_label_encoder", False)
        params["random_state"] = seed
        # XGBoost n'a pas class_weight, utiliser scale_pos_weight
        if "class_weight" in params:
            del params["class_weight"]
        return xgb.XGBClassifier(**params)

    if model_name == "catboost" and has_cb:
        import catboost as cb
        params = dict(hyperparams)
        params.setdefault("iterations", 300)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("depth", 6)
        params.setdefault("verbose", 0)
        params["random_seed"] = seed
        if "class_weight" in params:
            del params["class_weight"]
        return cb.CatBoostClassifier(**params)

    # Fallback LR
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        class_weight="balanced", max_iter=1000,
        C=1.0, random_state=seed
    )


# ─────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────

def _find_best_threshold(y, proba, lo=0.25, hi=0.80, step=0.01):
    best_t, best_f1 = 0.50, 0.0
    for t in np.arange(lo, hi + step, step):
        f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 3)


def _compute_metrics(y, proba, label="") -> dict:
    if len(np.unique(y)) < 2:
        return {"label": label, "error": "single_class"}
    t    = _find_best_threshold(y, proba)
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "label":     label,
        "auroc":     round(float(roc_auc_score(y, proba)), 4),
        "pr_auc":    round(float(average_precision_score(y, proba)), 4),
        "f1":        round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, pred, zero_division=0)), 4),
        "fp":        int(fp),
        "fn":        int(fn),
        "tp":        int(tp),
        "tn":        int(tn),
        "fpr":       round(float(fp / max(tn + fp, 1)), 4),
        "threshold": t,
    }


# ─────────────────────────────────────────────────────────────
# CV runner
# ─────────────────────────────────────────────────────────────

def _run_cv(label: str, df: pd.DataFrame, id_col: str, label_col: str,
            cfg: dict, n_folds: int, seed: int) -> tuple:
    """Retourne (metrics_dict, elapsed_seconds)."""
    t0 = time.time()
    _section(f"Run : {label}")

    feat = _extract_for_config(df, id_col, cfg)
    account_ids = list(feat[id_col])
    y   = df.groupby(id_col)[label_col].max().reindex(account_ids).values.astype(int)
    X   = _impute(feat.drop(columns=[id_col])).values
    n_feat = X.shape[1]

    _log(f"{len(y)} comptes | {n_feat} features | bots: {y.mean():.1%}")

    groups = pd.Series(account_ids)
    if groups.nunique() >= n_folds * 2:
        splitter = list(GroupKFold(n_splits=n_folds).split(X, y, groups=groups))
    else:
        splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed).split(X, y))

    oof = np.zeros(len(y))
    fold_aucs = []
    for fold, (tr, va) in enumerate(splitter, 1):
        m = _get_model_from_config(cfg, seed + fold)
        _, p = _fit_predict(m, X[tr], y[tr], X[va])
        oof[va] = p
        auc = roc_auc_score(y[va], p)
        fold_aucs.append(auc)
        _log(f"  Fold {fold}/{n_folds}  AUC={auc:.4f}")

    elapsed = round(time.time() - t0, 2)
    metrics  = _compute_metrics(y, oof, label=label)
    metrics["n_features"]       = n_feat
    metrics["n_accounts"]       = len(y)
    metrics["elapsed_seconds"]  = elapsed
    metrics["fold_aucs"]        = [round(a, 4) for a in fold_aucs]
    metrics["auroc_std"]        = round(float(np.std(fold_aucs)), 4)

    _log(f"OOF AUROC={metrics['auroc']:.4f}  "
         f"std={metrics['auroc_std']:.4f}  "
         f"F1={metrics['f1']:.4f}  "
         f"FP={metrics['fp']}  FN={metrics['fn']}  "
         f"({elapsed:.1f}s)")
    return metrics, elapsed


# ─────────────────────────────────────────────────────────────
# Calcul du delta et décision automatique
# ─────────────────────────────────────────────────────────────

def _compute_delta(baseline: dict, candidate: dict) -> dict:
    numeric_keys = ["auroc", "pr_auc", "f1", "precision", "recall",
                    "fp", "fn", "fpr", "elapsed_seconds", "n_features"]
    delta = {}
    for k in numeric_keys:
        b, c = baseline.get(k, 0), candidate.get(k, 0)
        delta[k] = round(float(c - b), 4)
    return delta


def _make_decision(baseline: dict, candidate: dict, delta: dict,
                   rules: dict) -> tuple:
    """
    Retourne (decision, reasons) selon RULES.md :
      keep        : gain > 0.01 AUROC, pas de régression FP, stable
      reject      : perte AUROC, ou FP augmente sans gain recall, ou très lent
      investigate : gain marginal, ou instabilité détectée
    """
    reasons   = []
    candidate_auroc_std = candidate.get("auroc_std", 0)
    time_ratio = (candidate.get("elapsed_seconds", 1) /
                  max(baseline.get("elapsed_seconds", 1), 0.1))

    # ── Critères de rejet (RULES.md §4) ──────────────────────
    if delta["auroc"] < -0.005:
        reasons.append(f"PERTE AUROC ({delta['auroc']:+.4f})")
        return "reject", reasons

    if delta["fp"] > 0 and delta["recall"] < 0.02:
        reasons.append(
            f"FP augmentent (+{delta['fp']}) sans gain recall significatif "
            f"(+{delta['recall']:+.4f})"
        )
        return "reject", reasons

    if candidate_auroc_std > 0.02:
        reasons.append(
            f"instabilite elevee (std={candidate_auroc_std:.4f} > 0.02)"
        )

    if time_ratio > 2.0:
        reasons.append(f"temps x{time_ratio:.1f} sans gain suffisant")
        if delta["auroc"] < 0.01:
            return "reject", reasons

    # ── Critère de validation (keep) ─────────────────────────
    if delta["auroc"] >= 0.01 and delta["fp"] <= 0 and candidate_auroc_std <= 0.02:
        reasons.append(
            f"gain AUROC={delta['auroc']:+.4f}, "
            f"FP={delta['fp']:+d}, "
            f"std={candidate_auroc_std:.4f}"
        )
        return "keep", reasons

    # ── Cas intermédiaires → investigate ─────────────────────
    if delta["auroc"] >= 0.005:
        reasons.append(f"gain modeste AUROC={delta['auroc']:+.4f} — necessaire validation sur dataset reel")
    elif delta["auroc"] >= 0 and delta["f1"] >= 0.005:
        reasons.append(f"gain F1={delta['f1']:+.4f} sans gain AUROC net — profil a evaluer")
    else:
        reasons.append(f"gain trop faible (AUROC={delta['auroc']:+.4f}) — laisser desactive")

    return "investigate", reasons


# ─────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────

def _print_comparison_table(baseline: dict, candidate: dict, delta: dict):
    cols = ["auroc", "pr_auc", "f1", "precision", "recall",
            "fp", "fn", "elapsed_seconds", "n_features"]
    print(f"\n  {'Metrique':<18} {'Baseline':>12} {'Candidat':>12} {'Delta':>12}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12}")
    for c in cols:
        b  = baseline.get(c, "—")
        cv = candidate.get(c, "—")
        d  = delta.get(c, "—")
        sign = "+" if isinstance(d, float) and d > 0 else ""
        # FP/FN : moins = mieux
        highlight = ""
        if c in ("fp", "fn", "elapsed_seconds") and isinstance(d, (int, float)):
            highlight = " (<= mieux)" if d > 0 else ""
        print(f"  {c:<18} {str(b):>12} {str(cv):>12} {f'{sign}{d}':>12}{highlight}")


def _print_decision(decision: str, reasons: list):
    icons = {"keep": "[KEEP]", "reject": "[REJECT]", "investigate": "[INVESTIGATE]"}
    icon  = icons.get(decision, "[?]")
    print(f"\n  DECISION : {icon}")
    for r in reasons:
        print(f"  • {r}")


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

def run_experiment(args) -> dict:
    t_global = time.time()
    _banner(f"EXPERIMENT RUNNER — BotOrNot  [{datetime.now():%Y-%m-%d %H:%M}]")

    # ── Vérification que baseline n'est pas modifiée ──────────
    baseline_path = args.baseline
    if not os.path.exists(baseline_path):
        sys.exit(f"Baseline introuvable : {baseline_path}")

    _log(f"Baseline  : {baseline_path}  [READ-ONLY]")
    _log(f"Experiment: {args.experiment}")

    baseline_cfg, candidate_cfg, experiment_overlay = _load_experiment_config(
        baseline_path, args.experiment
    )

    exp_meta = experiment_overlay.get("meta", {})
    exp_name = exp_meta.get("name", Path(args.experiment).stem)
    exp_desc = exp_meta.get("description", "")

    _log(f"Nom       : {exp_name}")
    if exp_desc:
        _log(f"Hypothese : {exp_desc}")

    # ── Affichage des différences ─────────────────────────────
    diffs = _diff_configs(baseline_cfg, candidate_cfg)
    if diffs:
        _section(f"Differences baseline → candidat ({len(diffs)} cles)")
        for d in diffs:
            print(f"  {d['key']}")
            print(f"    baseline  : {d['baseline']}")
            print(f"    candidat  : {d['candidate']}")
    else:
        _log("Aucune difference detectee — verifier le fichier experiment YAML")

    # ── Chargement données ────────────────────────────────────
    _section("Chargement des donnees")
    df = _load_file(args.train)
    _log(f"{len(df):,} lignes x {len(df.columns)} colonnes")

    id_col    = args.id_col    or _find_col(df, ID_PATTERNS) or "user_id"
    label_col = args.label_col or _find_col(df, LABEL_PATTERNS)
    if label_col is None:
        sys.exit("Colonne label introuvable. Utilisez --label-col.")

    seed    = args.seed
    n_folds = args.cv_folds

    # ── Run baseline (LECTURE SEULE de la config) ─────────────
    _banner("RUN 1/2 : BASELINE (golden_baseline)")
    baseline_metrics, baseline_time = _run_cv(
        "baseline", df, id_col, label_col, baseline_cfg, n_folds, seed
    )

    # ── Run candidat ──────────────────────────────────────────
    _banner("RUN 2/2 : CANDIDAT")
    candidate_metrics, candidate_time = _run_cv(
        exp_name, df, id_col, label_col, candidate_cfg, n_folds, seed
    )

    # ── Comparaison ───────────────────────────────────────────
    _banner("COMPARAISON")
    delta    = _compute_delta(baseline_metrics, candidate_metrics)
    decision, reasons = _make_decision(baseline_metrics, candidate_metrics, delta, {})

    _print_comparison_table(baseline_metrics, candidate_metrics, delta)
    _print_decision(decision, reasons)

    # ── Export ────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = os.path.join(args.out, f"{exp_name}_{timestamp}")

    report = {
        "generated_at":    datetime.now().isoformat(),
        "experiment_name": exp_name,
        "experiment_file": args.experiment,
        "baseline_file":   baseline_path,
        "description":     exp_desc,
        "hypothesis":      exp_meta.get("hypothesis", ""),
        "train":           args.train,
        "cv_folds":        n_folds,
        "seed":            seed,
        "elapsed_total_seconds": round(time.time() - t_global, 1),
        # Config diff
        "config_diff":     diffs,
        # Métriques
        "baseline":        baseline_metrics,
        "candidate":       candidate_metrics,
        "delta":           delta,
        # Décision
        "decision":        decision,
        "reasons":         reasons,
        # Règles appliquées (RULES.md)
        "rules_applied": {
            "min_auroc_gain_to_keep":   0.01,
            "max_auroc_std_to_keep":    0.02,
            "max_time_ratio_without_gain": 2.0,
        },
        # Recommandation actionnable
        "recommendation": {
            "keep": (
                f"Activer '{exp_name}' dans configs/features.yaml "
                f"(enabled: true) — gain AUROC={delta['auroc']:+.4f}"
            ),
            "reject": (
                f"Laisser '{exp_name}' desactive "
                "(ne satisfait pas les criteres RULES.md)"
            ),
            "investigate": (
                f"Tester '{exp_name}' sur un dataset reel avant de valider. "
                "Ne pas activer en production sans nouveau benchmark."
            ),
        }.get(decision, ""),
    }

    json_path = out_prefix + "_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # CSV résumé simple
    csv_path = out_prefix + "_summary.csv"
    rows = []
    for key in ["auroc", "pr_auc", "f1", "precision", "recall",
                "fp", "fn", "elapsed_seconds", "n_features"]:
        rows.append({
            "metric":    key,
            "baseline":  baseline_metrics.get(key),
            "candidate": candidate_metrics.get(key),
            "delta":     delta.get(key),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    _banner("RESULTAT FINAL")
    _log(f"Decision    : {decision.upper()}")
    _log(f"Recommandation : {report['recommendation']}")
    _log(f"Rapport JSON : {json_path}")
    _log(f"Rapport CSV  : {csv_path}")
    _log(f"Temps total  : {report['elapsed_total_seconds']:.1f}s")

    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Harness d'experimentation controlee BotOrNot (RULES.md)"
    )
    p.add_argument("--train",       required=True,
                   help="Dataset d'entrainement")
    p.add_argument("--test",        default=None,
                   help="Dataset de test (optionnel)")
    p.add_argument("--experiment",  required=True,
                   help="Fichier YAML de l'experience candidate")
    p.add_argument("--baseline",    default=GOLDEN_BASELINE_PATH,
                   help=f"Fichier de reference (defaut: {GOLDEN_BASELINE_PATH})")
    p.add_argument("--cv-folds",    type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default="artifacts/experiments",
                   help="Dossier de sortie des rapports")
    p.add_argument("--label-col",   default=None)
    p.add_argument("--id-col",      default=None)
    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
