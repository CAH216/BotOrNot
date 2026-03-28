#!/usr/bin/env python
"""
consensus_v2_compare.py — Benchmark V1 vs V2 du scoring de consensus
======================================================================
Compare les performances de ConsensusScorer (V1 moyenne pondérée)
et ConsensusScorerV2 (scoring enrichi) sur les mêmes données et folds.

Conforme à RULES.md :
  - La V2 est testée à côté de la V1, jamais à la place
  - La V2 ne sera activée que si le gain est stable (σ <= 0.02)
    et que les FP ne régressent pas
  - Rapport avant/après produit automatiquement

Usage :
    python scripts/consensus_v2_compare.py --train data/train.csv
    python scripts/consensus_v2_compare.py --train data/train.csv \\
        --n-seeds 5 --cv-folds 3 --out artifacts/consensus_v2
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
from src.inference.consensus import (
    ConsensusScorer, ConsensusScorerConfig,
    ConsensusScorerV2, ConsensusScorerV2Config,
)

SEP  = "=" * 72
SEP2 = "-" * 60

def _log(msg):     print(f"  [{datetime.now():%H:%M:%S}] {msg}")
def _banner(msg):  print(f"\n{SEP}\n  {msg}\n{SEP}")
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
    return float(best_t)


def _compute_metrics(y, proba, label="") -> dict:
    if len(np.unique(y)) < 2:
        return {m: 0.0 for m in ["auroc", "pr_auc", "f1", "precision",
                                  "recall", "fp", "fn"]}
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
# Simulation de block_probas modulaires depuis OOF
# ─────────────────────────────────────────────────────────────

def _simulate_block_probas(oof: np.ndarray, seed: int,
                           n_modules: int = 3) -> dict:
    """
    Simule plusieurs signaux modulaires à partir de l'OOF global.
    En production réelle, ce seraient les sorties de chaque module.
    Ici : OOF + bruit gaussien contrôlé pour chaque "module".
    """
    rng = np.random.default_rng(seed)
    modules = {}
    names   = ["tabular", "temporal", "text_basic", "structural"][:n_modules]
    noises  = [0.05, 0.07, 0.10, 0.04][:n_modules]
    for name, noise in zip(names, noises):
        arr = np.clip(oof + rng.normal(0, noise, len(oof)), 0, 1)
        modules[name] = arr
    return modules


# ─────────────────────────────────────────────────────────────
# Un run : retourne (V1_metrics, V2_metrics)
# ─────────────────────────────────────────────────────────────

def _single_run(df, id_col, label_col, n_folds, seed, model_name="lr",
                n_modules=3) -> tuple:
    feat = _build_features(df, id_col)
    ids  = list(feat[id_col])
    y    = df.groupby(id_col)[label_col].max().reindex(ids).values.astype(int)
    X    = _impute(feat.drop(columns=[id_col]).select_dtypes(include=[np.number])).values

    groups = pd.Series(ids)
    if groups.nunique() >= n_folds * 2:
        splitter = list(GroupKFold(n_splits=n_folds).split(X, y, groups=groups))
    else:
        splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed).split(X, y))

    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splitter, 1):
        m = _get_model(model_name, seed + fold)
        _, p = _fit_predict(m, X[tr], y[tr], X[va])
        oof[va] = p

    # ── V1 : probabilité brute OOF → ConsensusScorer ─────────
    block_probas = _simulate_block_probas(oof, seed, n_modules)
    v1_scorer    = ConsensusScorer()
    proba_v1     = v1_scorer.score(block_probas)

    # ── V2 : scoring enrichi (forcément activé pour le bench) ─
    v2_scorer = ConsensusScorerV2(enabled=True)
    v2_df     = v2_scorer.score(block_probas)
    proba_v2  = v2_df["proba"].values

    # Métriques
    m_v1 = _compute_metrics(y, proba_v1, "V1_weighted_avg")
    m_v2 = _compute_metrics(y, proba_v2, "V2_consensus")

    # Métriques de consensus (pour analyse)
    m_v1["mean_consensus_score"] = float(np.nan)
    m_v2["mean_consensus_score"] = float(v2_df["consensus_score"].mean())
    m_v2["mean_n_agree"]         = float(v2_df["n_agree"].mean())
    m_v2["mean_max_spread"]      = float(v2_df["max_spread"].mean())
    m_v2["mean_confidence"]      = float(v2_df["mean_confidence"].mean())

    return m_v1, m_v2


# ─────────────────────────────────────────────────────────────
# Agrégation multi-seeds
# ─────────────────────────────────────────────────────────────

def _aggregate(runs: list) -> dict:
    metrics = ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]
    result  = {}
    for m in metrics:
        vals = [r[m] for r in runs if m in r]
        arr  = np.array(vals, dtype=float)
        result[m] = {
            "mean":   round(float(np.mean(arr)), 4),
            "std":    round(float(np.std(arr)),  4),
            "min":    round(float(np.min(arr)),  4),
            "max":    round(float(np.max(arr)),  4),
        }
    # Extra
    for extra in ["mean_consensus_score", "mean_n_agree",
                  "mean_max_spread", "mean_confidence"]:
        vals = [r.get(extra, float("nan")) for r in runs]
        vals = [v for v in vals if not np.isnan(v)]
        result[extra] = round(float(np.mean(vals)), 4) if vals else float("nan")
    return result


# ─────────────────────────────────────────────────────────────
# Décision (RULES.md)
# ─────────────────────────────────────────────────────────────

def _decide(v1_agg: dict, v2_agg: dict) -> tuple:
    delta_auroc = v2_agg["auroc"]["mean"] - v1_agg["auroc"]["mean"]
    delta_fp    = v2_agg["fp"]["mean"]    - v1_agg["fp"]["mean"]
    sigma       = v2_agg["auroc"]["std"]
    reasons     = [
        f"delta_AUROC={delta_auroc:+.4f}",
        f"delta_FP={delta_fp:+.1f}",
        f"sigma(AUROC)={sigma:.4f}",
    ]
    if sigma > 0.02:
        reasons.append("UNSTABLE : sigma > 0.02 — ne pas activer")
        return "reject", reasons
    if delta_auroc < 0.005 and delta_fp >= 0:
        reasons.append("Gain negligeable — V1 reste superieure")
        return "reject", reasons
    if delta_fp > 0 and delta_auroc < 0.01:
        reasons.append("FP augmentent sans gain AUROC suffisant")
        return "reject", reasons
    if delta_auroc >= 0.01 and delta_fp <= 0 and sigma <= 0.02:
        reasons.append("Gain stable et non-regressif — activer V2")
        return "activate_v2", reasons
    reasons.append("Gain marginal — tester sur donnees reelles avant activation")
    return "investigate", reasons


# ─────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────

def _print_comparison(v1: dict, v2: dict, delta: dict):
    metrics = ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]
    print(f"\n  {'Metrique':<14} {'V1_mean':>10} {'V2_mean':>10} {'Delta':>10} {'V1_std':>8} {'V2_std':>8}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for m in metrics:
        b  = v1[m]["mean"]
        c  = v2[m]["mean"]
        d  = delta.get(m, 0)
        s1 = v1[m]["std"]
        s2 = v2[m]["std"]
        sign = "+" if isinstance(d, float) and d > 0 else ""
        print(f"  {m:<14} {b:>10.4f} {c:>10.4f} {f'{sign}{d:.4f}':>10} {s1:>8.4f} {s2:>8.4f}")

    print(f"\n  Métriques de consensus V2 :")
    for k in ["mean_consensus_score", "mean_n_agree", "mean_max_spread", "mean_confidence"]:
        v = v2.get(k, float("nan"))
        print(f"    {k:<28} = {v:.4f}" if not np.isnan(v) else f"    {k:<28} = N/A")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_consensus_compare(args) -> dict:
    t_global = time.time()
    _banner(f"CONSENSUS V1 vs V2 — BotOrNot  [{datetime.now():%Y-%m-%d %H:%M}]")
    _log("V1 : moyenne ponderee simple (stable, toujours active)")
    _log("V2 : scoring enrichi (n_agree + confidence + spread + structural_boost)")
    _log(f"RULES.md : V2 activee seulement si gain stable (delta >= 0.01, sigma <= 0.02)")

    df = _load_file(args.train)
    id_col    = args.id_col    or _find_col(df, ID_PATTERNS) or "user_id"
    label_col = args.label_col or _find_col(df, LABEL_PATTERNS)
    if label_col is None:
        sys.exit("Colonne label introuvable.")

    seeds = [args.seed + i * 17 for i in range(args.n_seeds)]
    _log(f"{len(df):,} lignes | {df[id_col].nunique()} comptes | "
         f"{args.n_seeds} seeds | {args.cv_folds} folds | {args.n_modules} modules")

    # ── Runs multi-seeds ─────────────────────────────────────
    _section(f"Runs ({args.n_seeds} seeds)")
    runs_v1, runs_v2 = [], []

    for i, seed in enumerate(seeds):
        t0 = time.time()
        _log(f"Seed {seed} ({i+1}/{args.n_seeds}) ...")
        m1, m2 = _single_run(df, id_col, label_col,
                              args.cv_folds, seed, "lr", args.n_modules)
        runs_v1.append(m1)
        runs_v2.append(m2)
        _log(f"  V1 AUROC={m1['auroc']:.4f}  V2 AUROC={m2['auroc']:.4f}  "
             f"({time.time()-t0:.1f}s)")

    # ── Agrégation ───────────────────────────────────────────
    agg_v1 = _aggregate(runs_v1)
    agg_v2 = _aggregate(runs_v2)

    delta = {}
    for m in ["auroc", "pr_auc", "f1", "precision", "recall", "fp", "fn"]:
        delta[m] = round(agg_v2[m]["mean"] - agg_v1[m]["mean"], 4)

    # ── Décision ─────────────────────────────────────────────
    decision, reasons = _decide(agg_v1, agg_v2)

    # ── Affichage ────────────────────────────────────────────
    _banner("RESULTATS V1 vs V2")
    _print_comparison(agg_v1, agg_v2, delta)

    _banner("DECISION (RULES.md)")
    icon = {
        "activate_v2": "[ACTIVER V2]",
        "reject":      "[REJETER V2 — garder V1]",
        "investigate": "[INVESTIGUER — tester sur donnees reelles]",
    }.get(decision, "[?]")
    print(f"\n  {icon}")
    for r in reasons:
        print(f"  -> {r}")

    if decision == "activate_v2":
        print(f"\n  ACTION : mettre enabled: true dans configs/golden_baseline.yaml")
        print(f"           sous la section consensus_v2:")
    else:
        print(f"\n  ACTION : maintenir consensus_v2.enabled: false")

    # ── Export ───────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out, f"consensus_compare_{timestamp}.json")
    csv_path  = os.path.join(args.out, f"consensus_compare_{timestamp}.csv")

    report = {
        "generated_at":     datetime.now().isoformat(),
        "elapsed_seconds":  round(time.time() - t_global, 1),
        "train":            args.train,
        "seeds":            seeds,
        "cv_folds":         args.cv_folds,
        "n_modules":        args.n_modules,
        "v1": agg_v1,
        "v2": agg_v2,
        "delta": delta,
        "decision":         decision,
        "reasons":          reasons,
        "rules_md": {
            "min_auroc_gain_to_activate": 0.01,
            "max_sigma_to_activate":      0.02,
            "fp_must_not_increase":       True,
        },
        "recommendation": {
            "activate_v2":  "Ajouter consensus_v2.enabled: true dans configs/golden_baseline.yaml",
            "reject":       "Garder consensus_v2.enabled: false (defaut)",
            "investigate":  "Tester sur dataset competitif reel avant toute activation",
        }.get(decision, ""),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    rows = [{"scorer": "V1", **{m: agg_v1[m]["mean"]  for m in delta.keys()},
             **{m+"_std": agg_v1[m]["std"] for m in delta.keys()}},
            {"scorer": "V2", **{m: agg_v2[m]["mean"]  for m in delta.keys()},
             **{m+"_std": agg_v2[m]["std"] for m in delta.keys()}},
            {"scorer": "delta", **delta}]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    elapsed = time.time() - t_global
    _banner("EXPORT")
    _log(f"Decision  : {decision}")
    _log(f"JSON      : {json_path}")
    _log(f"CSV       : {csv_path}")
    _log(f"Temps     : {elapsed:.1f}s")

    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark ConsensusScorer V1 vs V2 (RULES.md)"
    )
    p.add_argument("--train",      required=True)
    p.add_argument("--n-seeds",    type=int, default=5)
    p.add_argument("--cv-folds",   type=int, default=3)
    p.add_argument("--n-modules",  type=int, default=3,
                   help="Nb de modules a simuler (2-4, defaut: 3)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        default="artifacts/consensus_v2")
    p.add_argument("--label-col",  default=None)
    p.add_argument("--id-col",     default=None)
    args = p.parse_args()
    run_consensus_compare(args)


if __name__ == "__main__":
    main()
