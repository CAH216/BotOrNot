#!/usr/bin/env python
"""
benchmark.py — Benchmark interne automatique
=============================================
Compare 6 combinaisons de features sur un dataset fourni et exporte
un tableau de métriques complet.

Combinaisons benchmarkées :
  1. tabular only
  2. temporal only
  3. text only
  4. tabular + temporal
  5. tabular + temporal + text
  6. ensemble (moyenne des 3 scores indépendants)

Métriques exportées :
  AUROC, PR-AUC, F1, Precision, Recall, FP, FN, FPR, threshold

Usage :
    python scripts/benchmark.py --train data/train.csv
    python scripts/benchmark.py --train data/train.csv --model catboost --cv-folds 3
    python scripts/benchmark.py --train data/train.csv --out artifacts/benchmark_results
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
    f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Re-use helpers from run_baseline
sys.path.insert(0, str(Path(__file__).parent))
from run_baseline import (
    _load_file, _find_col,
    _make_tabular_features, _make_text_features, _make_temporal_features,
    _impute, _get_model, _fit_predict,
    ID_PATTERNS, LABEL_PATTERNS,
)

# ─────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────

COMBINATIONS = {
    "tabular_only":          ["tab"],
    "temporal_only":         ["tmp"],
    "text_only":             ["txt"],
    "tabular+temporal":      ["tab", "tmp"],
    "tabular+temporal+text": ["tab", "tmp", "txt"],
    "ensemble":              ["tab", "tmp", "txt"],  # blending spécial
}

SEP = "─" * 72


def _banner(msg):
    print(f"\n{SEP}")
    print(f"  {msg}")
    print(SEP)


def _log(msg):
    print(f"  [{datetime.now():%H:%M:%S}] {msg}")


# ─────────────────────────────────────────────────────────────────────────
# Extraction par bloc
# ─────────────────────────────────────────────────────────────────────────

def _extract_blocks(df: pd.DataFrame, id_col: str) -> dict:
    """Extrait les 3 blocs de features (tab, tmp, txt) indépendamment."""
    _log("Extracting tabular block…")
    tab_raw = _make_tabular_features(df, id_col)
    tab     = tab_raw.groupby(id_col).first().reset_index()

    _log("Extracting temporal block…")
    tmp = _make_temporal_features(df, id_col)

    _log("Extracting text block…")
    txt = _make_text_features(df, id_col)

    # Base des IDs
    ids = tab[[id_col]].copy()

    def _merge_block(name, block):
        merged = ids.merge(block, on=id_col, how="left")
        feat   = merged.drop(columns=[id_col]).select_dtypes(include=[np.number])
        # Supprimer colonnes avec > 95% NaN
        ok = [c for c in feat.columns if feat[c].isna().mean() < 0.95]
        return feat[ok]

    return {
        id_col: ids[id_col],
        "tab": _merge_block("tab", tab),
        "tmp": _merge_block("tmp", tmp),
        "txt": _merge_block("txt", txt),
    }


def _combine(blocks: dict, id_col: str, names: list) -> pd.DataFrame:
    """Fusionne les blocs sélectionnés en une seule matrice."""
    frames = [blocks[n] for n in names if n in blocks and blocks[n].shape[1] > 0]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    # Supprimer doublons de colonnes
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


# ─────────────────────────────────────────────────────────────────────────
# Calcul métriques
# ─────────────────────────────────────────────────────────────────────────

def _find_best_f1_threshold(y_true, proba):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.25, 0.80, 0.01):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 3)


def _compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = None) -> dict:
    if threshold is None:
        threshold = _find_best_f1_threshold(y_true, proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n_neg = tn + fp
    return {
        "auroc":     round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc":    round(float(average_precision_score(y_true, proba)), 4),
        "f1":        round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "fp":        int(fp),
        "fn":        int(fn),
        "tp":        int(tp),
        "tn":        int(tn),
        "fpr":       round(float(fp / max(n_neg, 1)), 4),
        "threshold": threshold,
        "n_features": 0,  # rempli après
    }


# ─────────────────────────────────────────────────────────────────────────
# CV sur une combinaison
# ─────────────────────────────────────────────────────────────────────────

def _run_combination(name: str, X: pd.DataFrame, y: np.ndarray,
                     groups: pd.Series, model_name: str,
                     n_folds: int, seed: int) -> dict:
    if X.shape[1] == 0:
        _log(f"  [{name}] ⚠️  Aucune feature — SKIPPED")
        return {"error": "no_features"}

    X_imp = _impute(X).values
    y_arr = y

    if groups is not None and groups.nunique() >= n_folds * 2:
        splitter = list(GroupKFold(n_splits=n_folds).split(X_imp, y_arr, groups=groups))
    else:
        splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed).split(X_imp, y_arr))

    oof = np.zeros(len(y_arr))
    for fold, (tr, va) in enumerate(splitter, 1):
        model = _get_model(model_name, seed + fold)
        _, proba = _fit_predict(model, X_imp[tr], y_arr[tr], X_imp[va])
        oof[va] = proba

    metrics = _compute_metrics(y_arr, oof)
    metrics["n_features"] = X.shape[1]
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Ensemble par blending des OOF
# ─────────────────────────────────────────────────────────────────────────

def _run_ensemble(blocks: dict, y: np.ndarray, groups: pd.Series,
                  model_name: str, n_folds: int, seed: int) -> dict:
    """Blend des probas OOF des 3 modèles indépendants."""
    block_names = ["tab", "tmp", "txt"]
    oofs = []

    for bname in block_names:
        X = blocks.get(bname, pd.DataFrame())
        if X.shape[1] == 0:
            continue
        X_imp = _impute(X).values
        if groups is not None and groups.nunique() >= n_folds * 2:
            splitter = list(GroupKFold(n_splits=n_folds).split(X_imp, y, groups=groups))
        else:
            splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                            random_state=seed).split(X_imp, y))
        oof = np.zeros(len(y))
        for fold, (tr, va) in enumerate(splitter, 1):
            model = _get_model(model_name, seed + fold)
            _, proba = _fit_predict(model, X_imp[tr], y[tr], X_imp[va])
            oof[va] = proba
        oofs.append(oof)

    if not oofs:
        return {"error": "no_blocks"}

    blended = np.mean(oofs, axis=0)
    metrics = _compute_metrics(y, blended)
    metrics["n_features"] = sum(blocks[b].shape[1] for b in block_names if b in blocks)
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Affichage du tableau résultats
# ─────────────────────────────────────────────────────────────────────────

def _print_results(results: dict) -> None:
    cols = ["auroc", "pr_auc", "f1", "precision", "recall", "fpr", "fp", "fn", "threshold", "n_features"]
    header = f"  {'combination':<28}" + "".join(f"{c:>12}" for c in cols)
    print(f"\n{SEP}")
    print("  RÉSULTATS DU BENCHMARK")
    print(SEP)
    print(header)
    print("  " + "-" * (26 + 12 * len(cols)))
    for comb, m in results.items():
        if "error" in m:
            print(f"  {comb:<28}  [SKIPPED — {m['error']}]")
            continue
        row = f"  {comb:<28}"
        for c in cols:
            v = m.get(c, "—")
            if isinstance(v, float):
                row += f"{v:>12.4f}"
            else:
                row += f"{str(v):>12}"
        print(row)
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def run_benchmark(args) -> dict:
    t_start = time.time()
    _banner(f"📊  BENCHMARK INTERNE — BotOrNot  [{datetime.now():%Y-%m-%d %H:%M}]")

    # Chargement
    _log(f"Train : {args.train}")
    df = _load_file(args.train)
    _log(f"  {len(df):,} lignes × {len(df.columns)} colonnes")

    id_col    = args.id_col    or _find_col(df, ID_PATTERNS)
    label_col = args.label_col or _find_col(df, LABEL_PATTERNS)

    if id_col is None:
        _log("⚠️  ID non détecté → index utilisé")
        df["_row_id"] = df.index.astype(str)
        id_col = "_row_id"

    if label_col is None:
        sys.exit("❌ Colonne label non trouvée. Utilisez --label-col.")

    _log(f"  ID: {id_col}  |  Label: {label_col}")

    # Labels par compte
    y_per_account = df.groupby(id_col)[label_col].max()

    # Extraction des blocs
    _banner("Extraction des features")
    blocks = _extract_blocks(df, id_col)

    # Réaligner l'ordre des comptes
    account_order = blocks[id_col].values
    y = y_per_account.reindex(account_order).values.astype(int)
    groups = pd.Series(account_order) if pd.Series(account_order).nunique() < len(account_order) else None

    _log(f"  {len(y):,} comptes  |  Positifs (bots): {y.mean():.1%}")
    _log(f"  Blocs — tab:{blocks['tab'].shape[1]}  tmp:{blocks['tmp'].shape[1]}  txt:{blocks['txt'].shape[1]} features")

    model_name = args.model
    try:
        import lightgbm; HAS_LGBM = True
    except ImportError:
        HAS_LGBM = False
    try:
        import catboost; HAS_CB = True
    except ImportError:
        HAS_CB = False

    if model_name == "lgbm" and not HAS_LGBM:
        model_name = "catboost" if HAS_CB else "lr"
    if model_name == "catboost" and not HAS_CB:
        model_name = "lr"

    _log(f"  Modèle : {model_name.upper()}  |  {args.cv_folds}-fold CV")

    # Benchmark
    _banner(f"Benchmark — 6 combinaisons × {args.cv_folds} folds")
    results = {}

    for comb_name, block_names in COMBINATIONS.items():
        t0 = time.time()
        _log(f"▶  {comb_name}…")

        if comb_name == "ensemble":
            m = _run_ensemble(blocks, y, groups, model_name, args.cv_folds, args.seed)
        else:
            X = _combine(blocks, id_col, block_names)
            m = _run_combination(comb_name, X, y, groups, model_name, args.cv_folds, args.seed)

        elapsed = time.time() - t0
        if "error" not in m:
            _log(f"   AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}  "
                 f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
                 f"FP={m['fp']}  FN={m['fn']}  ({elapsed:.1f}s)")
        results[comb_name] = m

    # Affichage
    _print_results(results)

    # Meilleure combinaison
    best = max(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda kv: kv[1]["auroc"],
        default=(None, None),
    )
    if best[0]:
        _log(f"🏆 Meilleure combinaison : {best[0]}  (AUROC={best[1]['auroc']:.4f})")

    total = time.time() - t_start

    # Export
    out_dir = os.path.dirname(args.out) if os.path.dirname(args.out) else "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. JSON complet
    full = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": round(total, 1),
        "train": args.train,
        "model": model_name,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "best_combination": best[0],
        "results": results,
    }
    json_path = args.out + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False, default=str)
    _log(f"JSON exporté → {json_path}")

    # 2. CSV synthèse
    rows = []
    for comb, m in results.items():
        if "error" in m:
            continue
        row = {"combination": comb}
        row.update(m)
        rows.append(row)

    if rows:
        csv_path = args.out + ".csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        _log(f"CSV exporté  → {csv_path}")

    _banner(f"✅  Benchmark terminé en {total:.1f}s")
    return full


def main():
    parser = argparse.ArgumentParser(
        description="📊 Benchmark interne — BotOrNot",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train",     required=True,  help="Fichier d'entraînement")
    parser.add_argument("--model",     default="lgbm",
                        choices=["lgbm", "catboost", "lr"],
                        help="Modèle (défaut: lgbm)")
    parser.add_argument("--cv-folds",  type=int, default=5,  help="Folds CV (défaut: 5)")
    parser.add_argument("--seed",      type=int, default=42, help="Seed (défaut: 42)")
    parser.add_argument("--out",       default="artifacts/benchmark",
                        help="Préfixe de sortie (défaut: artifacts/benchmark)")
    parser.add_argument("--label-col", default=None, help="Colonne label")
    parser.add_argument("--id-col",    default=None, help="Colonne ID compte")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
