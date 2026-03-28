#!/usr/bin/env python
"""
structural_v11_benchmark.py
============================
Benchmark V1 vs familles V1.1 du module structural (RULES.md §3).

Compare :
  - V1   (baseline stable — flags tous desactives)
  - F1   (source_v11 seulement)
  - F2   (batch_v11 seulement)
  - F3   (profile_v11 seulement)
  - F4   (template_v11 seulement)
  - Full (toutes les familles V1.1 activees)

Exporte JSON + CSV dans artifacts/structural_v11/

Usage :
    $env:PYTHONUTF8=1
    python scripts/structural_v11_benchmark.py --train data/train.csv
    python scripts/structural_v11_benchmark.py --train data/_dryrun_train.csv \\
        --n-seeds 3 --cv-folds 3 --out artifacts/structural_v11
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
    _load_file, _find_col, _build_features,
    _impute, _get_model, _fit_predict,
    ID_PATTERNS, LABEL_PATTERNS,
)

SEP  = "=" * 70
SEP2 = "-" * 58

def _log(m):    print(f"  [{datetime.now():%H:%M:%S}] {m}")
def _banner(m): print(f"\n{SEP}\n  {m}\n{SEP}")
def _section(m):print(f"\n{SEP2}\n  {m}")


# ─────────────────────────────────────────────────────────────
# Configs des familles V1.1
# ─────────────────────────────────────────────────────────────

FAMILIES = {
    "V1_baseline": {
        "source_v11_enabled":   False,
        "batch_v11_enabled":    False,
        "profile_v11_enabled":  False,
        "template_v11_enabled": False,
    },
    "F1_source": {
        "source_v11_enabled":   True,
        "batch_v11_enabled":    False,
        "profile_v11_enabled":  False,
        "template_v11_enabled": False,
    },
    "F2_batch": {
        "source_v11_enabled":   False,
        "batch_v11_enabled":    True,
        "profile_v11_enabled":  False,
        "template_v11_enabled": False,
    },
    "F3_profile": {
        "source_v11_enabled":   False,
        "batch_v11_enabled":    False,
        "profile_v11_enabled":  True,
        "template_v11_enabled": False,
    },
    "F4_template": {
        "source_v11_enabled":   False,
        "batch_v11_enabled":    False,
        "profile_v11_enabled":  False,
        "template_v11_enabled": True,
    },
    "Full_v11": {
        "source_v11_enabled":   True,
        "batch_v11_enabled":    True,
        "profile_v11_enabled":  True,
        "template_v11_enabled": True,
    },
}


# ─────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────

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
        "label":     label,
        "auroc":     round(float(roc_auc_score(y, proba)), 4),
        "pr_auc":    round(float(average_precision_score(y, proba)), 4),
        "f1":        round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, pred, zero_division=0)), 4),
        "fp":        int(fp),
        "fn":        int(fn),
        "threshold": round(t, 3),
    }


# ─────────────────────────────────────────────────────────────
# Un run : une seed, toutes les familles
# ─────────────────────────────────────────────────────────────

def _single_run(df, id_col, label_col, n_folds, seed,
                model_name="lr") -> dict:
    """Lance une CV sur toutes les familles V1.1 pour un seed donné."""
    # Construire les features V1 baseline (sans flag V1.1)
    feat_v1 = _build_features(df, id_col)
    ids     = list(feat_v1[id_col])
    y       = df.groupby(id_col)[label_col].max().reindex(ids).values.astype(int)
    X_v1    = _impute(feat_v1.drop(columns=[id_col]).select_dtypes(
                include=[np.number])).values

    groups = pd.Series(ids)
    if groups.nunique() >= n_folds * 2:
        splitter = list(GroupKFold(n_splits=n_folds).split(X_v1, y, groups=groups))
    else:
        splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed).split(X_v1, y))

    # OOF commun V1 baseline (base de référence)
    oof_v1 = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splitter, 1):
        m = _get_model(model_name, seed + fold)
        _, p = _fit_predict(m, X_v1[tr], y[tr], X_v1[va])
        oof_v1[va] = p

    results = {"V1_baseline": _metrics(y, oof_v1, "V1_baseline")}

    # Maintenant, pour les familles V1.1 : rebuild features avec flags
    # Note : _build_features ne passe pas encore de cfg — on enrichit manuellement
    # via extract_structural_features directement sur le df si structural dispo
    try:
        from src.features.structural import extract_structural_features, STRUCTURAL_COLS_V11
        from src.data.schema import AccountCols, PostCols
        has_id_col = AccountCols.ID in df.columns or id_col in df.columns

        for family_name, str_flags in FAMILIES.items():
            if family_name == "V1_baseline":
                continue

            cfg_family = {"structural": str_flags}
            # Extraire features structurelles V1.1 avec ce flag
            try:
                str_df = extract_structural_features(
                    accounts_df=df if has_id_col else None,
                    posts_df=None,
                    cfg=cfg_family,
                )
                # Merge avec feat_v1 (si dispo)
                if str_df is not None and not str_df.empty and AccountCols.ID in str_df.columns:
                    merged = feat_v1.merge(
                        str_df.add_suffix("_v11").rename(
                            columns={AccountCols.ID + "_v11": AccountCols.ID}),
                        on=AccountCols.ID, how="left",
                    )
                    X_fam = _impute(merged.drop(columns=[AccountCols.ID],
                                               errors="ignore")
                                    .select_dtypes(include=[np.number])).values
                else:
                    X_fam = X_v1
            except Exception:
                X_fam = X_v1

            # OOF famille
            oof_fam = np.zeros(len(y))
            for fold, (tr, va) in enumerate(splitter, 1):
                m = _get_model(model_name, seed + fold)
                _, p = _fit_predict(m, X_fam[tr], y[tr], X_fam[va])
                oof_fam[va] = p

            results[family_name] = _metrics(y, oof_fam, family_name)

    except ImportError as exc:
        _log(f"[WARN] extract_structural_features non disponible : {exc}")
        _log("       Seule la baseline V1 sera benchmarkee.")

    return results


# ─────────────────────────────────────────────────────────────
# Agrégation multi-seeds
# ─────────────────────────────────────────────────────────────

def _aggregate(all_seed_results: list, family_name: str) -> dict:
    metrics = ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]
    out = {}
    for m in metrics:
        vals = [r[family_name][m] for r in all_seed_results
                if family_name in r and m in r[family_name]]
        if not vals:
            out[m] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            continue
        arr = np.array(vals, dtype=float)
        out[m] = {
            "mean": round(float(np.mean(arr)), 4),
            "std":  round(float(np.std(arr)),  4),
            "min":  round(float(np.min(arr)),  4),
            "max":  round(float(np.max(arr)),  4),
        }
    return out


# ─────────────────────────────────────────────────────────────
# Décision par famille (RULES.md)
# ─────────────────────────────────────────────────────────────

def _decide_family(baseline: dict, candidate: dict, family: str) -> tuple:
    delta_auroc = candidate["auroc"]["mean"] - baseline["auroc"]["mean"]
    delta_fp    = candidate["fp"]["mean"]    - baseline["fp"]["mean"]
    sigma       = candidate["auroc"]["std"]
    reasons = [
        f"delta_AUROC={delta_auroc:+.4f}",
        f"delta_FP={delta_fp:+.1f}",
        f"sigma(AUROC)={sigma:.4f}",
    ]
    if sigma > 0.02:
        return "reject", reasons + ["UNSTABLE (sigma > 0.02)"]
    if delta_fp > 0 and delta_auroc < 0.01:
        return "reject", reasons + ["FP augmentent sans gain suffisant"]
    if delta_auroc < 0.005:
        return "reject", reasons + ["Gain negligeable (< 0.005)"]
    if delta_auroc >= 0.01 and delta_fp <= 0 and sigma <= 0.02:
        return "activate", reasons + [f"Gain stable — activer {family}"]
    return "investigate", reasons + ["Tester sur dataset reel avant activation"]


# ─────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────

def _print_table(agg_results: dict, ref_name="V1_baseline"):
    ref = agg_results[ref_name]
    metrics = ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]
    print(f"\n  {'Famille':<16} {'AUROC':>8} {'PR-AUC':>8} {'F1':>8} "
          f"{'Prec':>8} {'Rec':>8} {'FP':>6} {'FN':>6} {'dAUROC':>9} {'sigma':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*9} {'-'*8}")
    for name, agg in agg_results.items():
        d   = agg["auroc"]["mean"] - ref["auroc"]["mean"] if name != ref_name else 0.0
        sig = agg["auroc"]["std"]
        sign = "+" if d > 0 else ""
        print(
            f"  {name:<16} {agg['auroc']['mean']:>8.4f} {agg['pr_auc']['mean']:>8.4f} "
            f"{agg['f1']['mean']:>8.4f} {agg['precision']['mean']:>8.4f} "
            f"{agg['recall']['mean']:>8.4f} {int(agg['fp']['mean']):>6d} "
            f"{int(agg['fn']['mean']):>6d} {f'{sign}{d:.4f}':>9} {sig:>8.4f}"
        )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_structural_v11_benchmark(args) -> dict:
    t0 = time.time()
    _banner(f"STRUCTURAL V1.1 BENCHMARK — [{datetime.now():%Y-%m-%d %H:%M}]")
    _log("Familles : V1_baseline | F1_source | F2_batch | F3_profile | F4_template | Full_v11")
    _log("RULES.md : activation si delta_AUROC >= 0.01, sigma <= 0.02, FP stable")

    df        = _load_file(args.train)
    id_col    = args.id_col    or _find_col(df, ID_PATTERNS) or "user_id"
    label_col = args.label_col or _find_col(df, LABEL_PATTERNS)
    if label_col is None:
        sys.exit("Colonne label introuvable.")

    seeds = [args.seed + i * 17 for i in range(args.n_seeds)]
    _log(f"{len(df):,} lignes | {df[id_col].nunique()} comptes | "
         f"{args.n_seeds} seeds | {args.cv_folds} folds")

    # ── Runs multi-seeds ─────────────────────────────────────
    _section(f"Runs ({args.n_seeds} seeds × {args.cv_folds} folds)")
    all_results = []
    for i, seed in enumerate(seeds, 1):
        t_s = time.time()
        _log(f"Seed {seed} ({i}/{args.n_seeds}) ...")
        res = _single_run(df, id_col, label_col, args.cv_folds, seed)
        all_results.append(res)
        baseline_auroc = res.get("V1_baseline", {}).get("auroc", 0.0)
        _log(f"  V1={baseline_auroc:.4f} | {time.time()-t_s:.1f}s")

    # ── Agrégation ───────────────────────────────────────────
    families_found = list(all_results[0].keys()) if all_results else list(FAMILIES.keys())
    agg = {fam: _aggregate(all_results, fam) for fam in families_found}

    # ── Affichage ────────────────────────────────────────────
    _banner("RESULTATS V1 vs FAMILLES V1.1")
    _print_table(agg)

    # ── Décisions ────────────────────────────────────────────
    _banner("DECISIONS PAR FAMILLE (RULES.md)")
    decisions = {}
    print(f"\n  {'Famille':<16} {'Decision':<14} Raisons")
    print(f"  {'-'*16} {'-'*14} {'-'*40}")
    for fam in families_found:
        if fam == "V1_baseline":
            continue
        dec, reasons = _decide_family(agg["V1_baseline"], agg[fam], fam)
        decisions[fam] = {"decision": dec, "reasons": reasons}
        icon = {"activate": "[ACTIVER]", "reject": "[REJETER]",
                "investigate": "[INVESTIGUER]"}.get(dec, "[?]")
        print(f"  {fam:<16} {icon:<14} {reasons[0]}, {reasons[1]}")

    print("\n  Instructions d'activation (si decision = activate) :")
    print("  → Mettre le flag a true dans configs/features.yaml, section structural")

    # ── Export ───────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out, f"structural_v11_{ts}.json")
    csv_path  = os.path.join(args.out, f"structural_v11_{ts}.csv")

    report = {
        "generated_at":    datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "train":           args.train,
        "seeds":           seeds,
        "cv_folds":        args.cv_folds,
        "aggregated":      agg,
        "decisions":       decisions,
        "rules": {
            "min_auroc_gain":        0.01,
            "max_sigma":             0.02,
            "fp_must_not_increase":  True,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    rows = []
    for fam in families_found:
        row = {"famille": fam}
        for m in ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]:
            row[m]          = agg[fam][m]["mean"]
            row[m + "_std"] = agg[fam][m]["std"]
        if fam != "V1_baseline":
            row["decision"] = decisions.get(fam, {}).get("decision", "")
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    _banner("EXPORT")
    _log(f"JSON      : {json_path}")
    _log(f"CSV       : {csv_path}")
    _log(f"Temps total : {elapsed:.1f}s")
    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark structural V1 vs V1.1 familles (RULES.md)"
    )
    p.add_argument("--train",      required=True)
    p.add_argument("--n-seeds",    type=int, default=5)
    p.add_argument("--cv-folds",   type=int, default=3)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        default="artifacts/structural_v11")
    p.add_argument("--label-col",  default=None)
    p.add_argument("--id-col",     default=None)
    args = p.parse_args()
    run_structural_v11_benchmark(args)


if __name__ == "__main__":
    main()
