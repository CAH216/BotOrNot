#!/usr/bin/env python
"""
stability_check.py — Analyse de stabilité multi-seeds / multi-splits
======================================================================
Teste si un gain de performance est réel ou dû au hasard du split.

Pour chaque variante déclarée, on lance N séquences de seeds et on
calcule moyenne, écart-type, min, max des métriques principales.

Classification finale (conforme à RULES.md §4) :
  stable_gain    : delta AUROC >= 0.01, sigma <= 0.02, delta_fp <= 0
  unstable_gain  : delta AUROC >= 0.01, mais sigma > 0.02 (risque en compétition)
  no_gain        : delta AUROC < 0.01 (gain trop faible ou nul)

Usage :
    # Tester la baseline seule (calibrer la variance)
    python scripts/stability_check.py --train data/train.csv

    # Comparer baseline vs une variante
    python scripts/stability_check.py --train data/train.csv \\
        --variants configs/experiments/example_xgb_model.yaml

    # Plusieurs variantes, n seeds
    python scripts/stability_check.py --train data/train.csv \\
        --variants configs/experiments/example_xgb_model.yaml \\
                   configs/experiments/example_text_model_enabled.yaml \\
        --n-seeds 5 --cv-folds 3 --out artifacts/stability
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
from experiment_runner import (
    _load_yaml, _deep_merge, _get_model_from_config, _extract_for_config,
)

SEP  = "=" * 72
SEP2 = "-" * 60

METRICS = ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]

# Critères de classification RULES.md §4
MIN_AUROC_GAIN     = 0.01   # gain minimal pour "gain"
MAX_SIGMA_STABLE   = 0.02   # sigma max pour "stable"
MAX_FP_INCREASE    = 0      # FP ne doit pas augmenter


def _log(msg):    print(f"  [{datetime.now():%H:%M:%S}] {msg}")
def _banner(msg): print(f"\n{SEP}\n  {msg}\n{SEP}")
def _section(msg): print(f"\n{SEP2}\n  {msg}")


# ─────────────────────────────────────────────────────────────
# Métriques avec seuil optimal
# ─────────────────────────────────────────────────────────────

def _best_threshold(y, proba):
    best_t, best_f1 = 0.50, 0.0
    for t in np.arange(0.25, 0.80, 0.02):
        f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 3)


def _compute_metrics(y, proba) -> dict:
    if len(np.unique(y)) < 2:
        return {m: float("nan") for m in METRICS}
    t    = _best_threshold(y, proba)
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc":     float(roc_auc_score(y, proba)),
        "pr_auc":    float(average_precision_score(y, proba)),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "fp":        float(fp),
        "fn":        float(fn),
        "threshold": t,
    }


# ─────────────────────────────────────────────────────────────
# Un run = 1 seed × 1 config
# ─────────────────────────────────────────────────────────────

def _single_run(df, id_col, label_col, cfg, n_folds, seed) -> dict:
    feat  = _extract_for_config(df, id_col, cfg)
    ids   = list(feat[id_col])
    y     = df.groupby(id_col)[label_col].max().reindex(ids).values.astype(int)
    X     = _impute(feat.drop(columns=[id_col])).values

    groups = pd.Series(ids)
    if groups.nunique() >= n_folds * 2:
        splitter = list(GroupKFold(n_splits=n_folds).split(X, y, groups=groups))
    else:
        splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed).split(X, y))
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splitter, 1):
        m = _get_model_from_config(cfg, seed + fold)
        _, p = _fit_predict(m, X[tr], y[tr], X[va])
        oof[va] = p

    return _compute_metrics(y, oof)


# ─────────────────────────────────────────────────────────────
# Agrégation statistique sur N seeds
# ─────────────────────────────────────────────────────────────

def _aggregate(runs: list) -> dict:
    """Calcule mean / std / min / max / median pour chaque métrique."""
    result = {}
    for m in METRICS:
        vals = [r[m] for r in runs if not np.isnan(r.get(m, float("nan")))]
        if not vals:
            result[m] = {"mean": None, "std": None, "min": None,
                         "max": None, "median": None}
            continue
        arr = np.array(vals)
        result[m] = {
            "mean":   round(float(np.mean(arr)),   4),
            "std":    round(float(np.std(arr)),    4),
            "min":    round(float(np.min(arr)),    4),
            "max":    round(float(np.max(arr)),    4),
            "median": round(float(np.median(arr)), 4),
        }
    return result


# ─────────────────────────────────────────────────────────────
# Classification (RULES.md §4)
# ─────────────────────────────────────────────────────────────

def _classify(baseline_agg: dict, candidate_agg: dict) -> tuple:
    """
    Returns (classification, reasons_list).
    baseline_agg == candidate_agg → classify baseline itself as reference.
    """
    b_auroc  = baseline_agg["auroc"]["mean"]  or 0
    c_auroc  = candidate_agg["auroc"]["mean"] or 0
    c_sigma  = candidate_agg["auroc"]["std"]  or 0
    b_fp     = baseline_agg["fp"]["mean"]     or 0
    c_fp     = candidate_agg["fp"]["mean"]    or 0
    delta_auroc = c_auroc - b_auroc
    delta_fp    = c_fp - b_fp

    reasons = [
        f"delta_AUROC={delta_auroc:+.4f}",
        f"sigma_AUROC={c_sigma:.4f}",
        f"delta_FP={delta_fp:+.1f}",
    ]

    # Instabilité (RULES.md §4 : σ > 0.02)
    if c_sigma > MAX_SIGMA_STABLE:
        reasons.append(f"INSTABLE : sigma={c_sigma:.4f} > {MAX_SIGMA_STABLE}")
        if delta_auroc >= MIN_AUROC_GAIN:
            return "unstable_gain", reasons
        return "no_gain", reasons

    # Gain insuffisant
    if delta_auroc < MIN_AUROC_GAIN:
        reasons.append(f"Gain trop faible : {delta_auroc:+.4f} < {MIN_AUROC_GAIN}")
        return "no_gain", reasons

    # FP augmentent sans justification
    if delta_fp > MAX_FP_INCREASE:
        reasons.append(f"FP augmentent : +{delta_fp:.0f}")
        return "unstable_gain", reasons

    # Gain stable et net
    reasons.append("Gain stable et non-regressif")
    return "stable_gain", reasons


# ─────────────────────────────────────────────────────────────
# Affichage tableau de résultats
# ─────────────────────────────────────────────────────────────

def _print_stats_row(name: str, agg: dict, col_width: int = 25):
    """Imprime une ligne : mean ± std  [min, max]"""
    w = col_width
    print(f"\n  {'Config':<{w}} {'AUROC':>10} {'PR-AUC':>10} "
          f"{'F1':>10} {'Prec':>10} {'Rec':>10} {'FP':>8} {'FN':>8}")
    print(f"  {'-'*w} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    def _fmt(stats):
        if stats["mean"] is None:
            return "      N/A"
        return f"{stats['mean']:>7.4f}"

    row = f"  {name:<{w}}"
    for m in ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]:
        row += f" {_fmt(agg[m]):>10}"
    print(row)

    # Ligne std
    def _fmt_std(stats):
        if stats["std"] is None:
            return "      N/A"
        return f"+-{stats['std']:.4f}"
    row2 = f"  {'  (std)':>{w}}"
    for m in ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]:
        row2 += f" {_fmt_std(agg[m]):>10}"
    print(row2)



def _print_summary_table(results: list):
    icons = {
        "stable_gain":   "[STABLE_GAIN]  ",
        "unstable_gain": "[UNSTABLE_GAIN]",
        "no_gain":       "[NO_GAIN]      ",
        "reference":     "[REFERENCE]    ",
    }
    print(f"\n  {'Config':<30} {'Classif':>16} {'AUROC_mean':>12} "
          f"{'AUROC_std':>12} {'delta_AUROC':>13} {'delta_FP':>10}")
    print(f"  {'-'*30} {'-'*16} {'-'*12} {'-'*12} {'-'*13} {'-'*10}")
    ref_auroc = results[0]["agg"]["auroc"]["mean"] if results else 0
    for r in results:
        m      = r["agg"]["auroc"]
        delta  = round((m["mean"] or 0) - (ref_auroc or 0), 4)
        d_fp   = round((r["agg"]["fp"]["mean"] or 0) - (results[0]["agg"]["fp"]["mean"] or 0), 1)
        icon   = icons.get(r.get("classification", "reference"), "[?]")
        sign   = "+" if delta >= 0 else ""
        print(f"  {r['name']:<30} {icon:>16} {m['mean']:>12.4f} "
              f"{m['std']:>12.4f} {f'{sign}{delta}':>13} {f'{d_fp:+.0f}':>10}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_stability(args) -> list:
    t_global = time.time()
    _banner(f"STABILITY CHECK — BotOrNot  [{datetime.now():%Y-%m-%d %H:%M}]")

    # ── Chargement données ────────────────────────────────────
    df = _load_file(args.train)
    id_col    = args.id_col    or _find_col(df, ID_PATTERNS) or "user_id"
    label_col = args.label_col or _find_col(df, LABEL_PATTERNS)
    if label_col is None:
        sys.exit("Colonne label introuvable. Utilisez --label-col.")

    _log(f"{len(df):,} lignes | id={id_col} | label={label_col}")

    # ── Seeds à tester ────────────────────────────────────────
    base_seed = args.seed
    seeds     = [base_seed + i * 17 for i in range(args.n_seeds)]
    _log(f"Seeds : {seeds}  |  CV folds : {args.cv_folds}  |  "
         f"N runs par variante : {args.n_seeds}")

    # ── Charger baseline ──────────────────────────────────────
    baseline_path = args.baseline
    if not os.path.exists(baseline_path):
        sys.exit(f"Baseline introuvable : {baseline_path}")
    baseline_cfg = _load_yaml(baseline_path)

    # ── Construire la liste des variantes à tester ────────────
    # Premier = toujours la baseline (référence)
    variants = [{"name": "golden_baseline", "cfg": baseline_cfg, "file": baseline_path}]

    for vpath in (args.variants or []):
        if not os.path.exists(vpath):
            _log(f"  Variante introuvable : {vpath} — ignorée")
            continue
        vname    = Path(vpath).stem
        exp_cfg  = _load_yaml(vpath)
        cand_cfg = _deep_merge(baseline_cfg, exp_cfg)
        variants.append({"name": vname, "cfg": cand_cfg, "file": vpath})

    _log(f"Variantes : {[v['name'] for v in variants]}")

    # ── Run : N seeds × M variantes ───────────────────────────
    all_run_results = {}
    for v in variants:
        _section(f"Variante : {v['name']}")
        runs = []
        for s in seeds:
            t0 = time.time()
            try:
                metrics = _single_run(df, id_col, label_col, v["cfg"],
                                      args.cv_folds, s)
                runs.append(metrics)
                _log(f"  seed={s:<6}  AUROC={metrics['auroc']:.4f}  "
                     f"F1={metrics['f1']:.4f}  FP={metrics['fp']:.0f}  "
                     f"({time.time()-t0:.1f}s)")
            except Exception as e:
                _log(f"  seed={s:<6}  ERREUR : {e}")
        all_run_results[v["name"]] = {"config": v, "runs": runs}

    # ── Agrégation + classification ───────────────────────────
    _banner("RESULTATS STATISTIQUES")
    baseline_agg = _aggregate(all_run_results["golden_baseline"]["runs"])

    compiled = []
    for vname, data in all_run_results.items():
        agg = _aggregate(data["runs"])
        if vname == "golden_baseline":
            cls, reasons = "reference", ["Variante de reference"]
        else:
            cls, reasons = _classify(baseline_agg, agg)
        compiled.append({
            "name":            vname,
            "file":            data["config"]["file"],
            "n_runs":          len(data["runs"]),
            "agg":             agg,
            "classification":  cls,
            "reasons":         reasons,
            "raw_runs":        data["runs"],
        })
        _print_stats_row(vname, agg)
        print(f"  Classification : [{cls.upper()}]")
        for r in reasons:
            print(f"    -> {r}")

    # ── Tableau récapitulatif ─────────────────────────────────
    _banner("TABLEAU COMPARATIF")
    _print_summary_table(compiled)

    # ── Recommandations ───────────────────────────────────────
    _banner("RECOMMANDATIONS (RULES.md)")
    for r in compiled[1:]:  # skip baseline
        cls = r["classification"]
        if cls == "stable_gain":
            print(f"  [VALIDER]     {r['name']} — gain stable, peut etre active")
        elif cls == "unstable_gain":
            print(f"  [PRUDENCE]    {r['name']} — gain present mais instable (sigma > {MAX_SIGMA_STABLE})")
            print(f"                -> Tester sur dataset reel avant d'activer")
        else:
            print(f"  [REJETER]     {r['name']} — pas de gain significatif, laisser desactive")

    # ── Export ────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.out, f"stability_{timestamp}.json")
    csv_path    = os.path.join(args.out, f"stability_{timestamp}.csv")

    # JSON complet
    report = {
        "generated_at":     datetime.now().isoformat(),
        "elapsed_seconds":  round(time.time() - t_global, 1),
        "train":            args.train,
        "baseline":         baseline_path,
        "n_seeds":          args.n_seeds,
        "seeds":            seeds,
        "cv_folds":         args.cv_folds,
        "classification_thresholds": {
            "min_auroc_gain_for_gain":     MIN_AUROC_GAIN,
            "max_sigma_for_stable":        MAX_SIGMA_STABLE,
            "max_fp_increase_for_stable":  MAX_FP_INCREASE,
        },
        "results": [
            {k: v for k, v in r.items() if k != "raw_runs"}
            for r in compiled
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # CSV résumé (une ligne par variante × métrique)
    rows = []
    for r in compiled:
        base_row = {
            "name":           r["name"],
            "n_runs":         r["n_runs"],
            "classification": r["classification"],
        }
        for m in METRICS:
            for stat in ("mean", "std", "min", "max", "median"):
                base_row[f"{m}_{stat}"] = r["agg"][m][stat]
        rows.append(base_row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    _banner(f"EXPORT")
    total = time.time() - t_global
    _log(f"JSON : {report_path}")
    _log(f"CSV  : {csv_path}")
    _log(f"Temps total : {total:.1f}s  ({total/60:.1f} min)")

    return compiled


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Analyse de stabilite multi-seeds BotOrNot (RULES.md)"
    )
    p.add_argument("--train",       required=True,
                   help="Dataset d'entrainement")
    p.add_argument("--variants",    nargs="*", default=[],
                   help="Fichiers YAML des variantes candidates (0 = baseline seule)")
    p.add_argument("--baseline",    default="configs/golden_baseline.yaml")
    p.add_argument("--n-seeds",     type=int,   default=5,
                   help="Nombre de seeds distincts a tester (defaut: 5)")
    p.add_argument("--cv-folds",    type=int,   default=3,
                   help="Folds CV par seed (defaut: 3)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Seed de base (les suivants sont +17, +34, ...)")
    p.add_argument("--out",         default="artifacts/stability")
    p.add_argument("--label-col",   default=None)
    p.add_argument("--id-col",      default=None)
    args = p.parse_args()
    run_stability(args)


if __name__ == "__main__":
    main()
