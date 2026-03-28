#!/usr/bin/env python
"""
ablation.py — Ablation study automatique
=========================================
Mesure l'apport individuel de chaque module de features sur les métriques
de classification, en comparaison avec l'ensemble complet.

Modules testés :
  - tabular only
  - temporal only
  - text_basic only
  - text_model (TF-IDF)
  - structural (si colonnes disponibles)
  - tabular + temporal
  - tabular + temporal + text_basic
  - ensemble complet (blending)
  - ablation tabular       (ensemble sans tabular)
  - ablation temporal      (ensemble sans temporal)
  - ablation text_basic    (ensemble sans text_basic)

Métriques : AUROC, PR-AUC, F1, Precision, Recall, FP, FN, FPR, threshold

Usage :
    python scripts/ablation.py --train data/train.csv
    python scripts/ablation.py --train data/train.csv --cv-folds 3 --out artifacts/ablation
"""
import sys, os, time, json, argparse, warnings
from pathlib import Path
from datetime import datetime
from itertools import combinations

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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

SEP = "─" * 78


def _banner(msg):
    print(f"\n{SEP}")
    print(f"  {msg}")
    print(SEP)


def _log(msg):
    print(f"  [{datetime.now():%H:%M:%S}] {msg}")


# ─────────────────────────────────────────────────────────────
# Extraction des blocs (même logique que benchmark.py)
# ─────────────────────────────────────────────────────────────

def _extract_all_blocks(df: pd.DataFrame, id_col: str) -> dict:
    _log("Extracting tabular block…")
    tab_raw = _make_tabular_features(df, id_col)
    tab     = tab_raw.groupby(id_col).first().reset_index()

    _log("Extracting temporal block…")
    tmp = _make_temporal_features(df, id_col)

    _log("Extracting text block…")
    txt = _make_text_features(df, id_col)

    # TF-IDF text model (léger)
    _log("Extracting text_model block (TF-IDF)…")
    text_col = _find_col(df, ["text", "content", "tweet", "post", "body"])
    tfidf_feat = None
    if text_col and df[text_col].notna().mean() > 0.1:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            texts_per_account = (
                df.groupby(id_col)[text_col]
                  .apply(lambda x: " ".join(x.fillna("").astype(str)))
                  .reset_index()
            )
            texts_per_account.columns = [id_col, "_joined_text"]
            tfidf  = TfidfVectorizer(max_features=300, sublinear_tf=True, min_df=2)
            matrix = tfidf.fit_transform(texts_per_account["_joined_text"])
            n_components = min(30, matrix.shape[1] - 1)
            svd    = TruncatedSVD(n_components=n_components, random_state=42)
            lsa    = svd.fit_transform(matrix)
            tfidf_feat = pd.DataFrame(
                lsa,
                columns=[f"tfidf_svd_{i}" for i in range(lsa.shape[1])]
            )
            tfidf_feat[id_col] = texts_per_account[id_col].values
            _log(f"  text_model : {lsa.shape[1]} features TF-IDF/SVD")
        except Exception as e:
            _log(f"  text_model : échec ({e})")

    # Structural (si colonnes source / client présentes)
    _log("Checking structural signals…")
    struct_feat = None
    struct_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ["source", "client", "app", "device", "platform"]
    )]
    if struct_cols:
        try:
            g = df[[id_col] + struct_cols].copy()
            src_col = struct_cols[0]
            agg = g.groupby(id_col).agg(
                struct_n_sources=(src_col, "nunique"),
                struct_most_common=(src_col, lambda x: x.value_counts().index[0] if len(x) else ""),
            ).reset_index()
            struct_feat = agg
            _log(f"  structural : {len(struct_cols)} colonnes source détectées")
        except Exception as e:
            _log(f"  structural : échec ({e})")

    # Base IDs
    ids = tab[[id_col]].copy()

    def _merge(block, name):
        if block is None or (hasattr(block, "shape") and block.shape[1] == 0):
            return pd.DataFrame()
        m = ids.merge(block, on=id_col, how="left")
        feat = m.drop(columns=[id_col]).select_dtypes(include=[np.number])
        ok = [c for c in feat.columns if feat[c].isna().mean() < 0.95]
        return feat[ok]

    return {
        id_col:     ids[id_col],
        "tabular":  _merge(tab, "tabular"),
        "temporal": _merge(tmp, "temporal"),
        "text_basic": _merge(txt, "text_basic"),
        "text_model": _merge(tfidf_feat, "text_model") if tfidf_feat is not None else pd.DataFrame(),
        "structural": _merge(struct_feat, "structural") if struct_feat is not None else pd.DataFrame(),
    }


# ─────────────────────────────────────────────────────────────
# Métriques + seuil optimal
# ─────────────────────────────────────────────────────────────

def _best_threshold(y, proba):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.25, 0.80, 0.01):
        f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 3)


def _metrics(y, proba, label: str, n_feat: int) -> dict:
    if len(np.unique(y)) < 2:
        return {"label": label, "error": "single_class"}
    t = _best_threshold(y, proba)
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "label":     label,
        "n_feat":    n_feat,
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
# CV sur une combinaison de blocs
# ─────────────────────────────────────────────────────────────

def _cv_on_blocks(blocks: dict, id_col: str, block_names: list,
                  y: np.ndarray, groups,
                  model_name: str, n_folds: int, seed: int,
                  label: str, blend: bool = False) -> dict:
    if blend:
        # Blending : entraîner un modèle par bloc et moyenner
        oofs = []
        for bname in block_names:
            X = blocks.get(bname, pd.DataFrame())
            if X.shape[1] == 0:
                continue
            X_imp = _impute(X).values
            splitter = _make_splitter(X_imp, y, groups, n_folds, seed)
            oof = np.zeros(len(y))
            for fold, (tr, va) in enumerate(splitter, 1):
                m = _get_model(model_name, seed + fold)
                _, p = _fit_predict(m, X_imp[tr], y[tr], X_imp[va])
                oof[va] = p
            oofs.append(oof)
        if not oofs:
            return {"label": label, "error": "no_blocks"}
        proba = np.mean(oofs, axis=0)
        n_feat = sum(blocks[b].shape[1] for b in block_names if b in blocks)
    else:
        # Concaténation simple
        frames = [blocks[b] for b in block_names
                  if b in blocks and blocks[b].shape[1] > 0]
        if not frames:
            return {"label": label, "error": "no_features"}
        X = pd.concat(frames, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
        n_feat = X.shape[1]
        X_imp = _impute(X).values
        splitter = _make_splitter(X_imp, y, groups, n_folds, seed)
        proba = np.zeros(len(y))
        for fold, (tr, va) in enumerate(splitter, 1):
            m = _get_model(model_name, seed + fold)
            _, p = _fit_predict(m, X_imp[tr], y[tr], X_imp[va])
            proba[va] = p

    return _metrics(y, proba, label, n_feat)


def _make_splitter(X, y, groups, n_folds, seed):
    if groups is not None and groups.nunique() >= n_folds * 2:
        return list(GroupKFold(n_splits=n_folds).split(X, y, groups=groups))
    return list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=seed).split(X, y))


# ─────────────────────────────────────────────────────────────
# Affichage résultats
# ─────────────────────────────────────────────────────────────

def _print_table(results: list) -> None:
    cols = ["auroc", "pr_auc", "f1", "precision", "recall", "fpr", "fp", "fn", "n_feat", "threshold"]
    print(f"\n{SEP}")
    print("  ABLATION STUDY — RÉSULTATS")
    print(SEP)
    hdr = f"  {'configuration':<32}" + "".join(f"{c:>10}" for c in cols)
    print(hdr)
    print("  " + "-" * (30 + 10 * len(cols)))
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<32}  [SKIPPED — {r['error']}]")
            continue
        row = f"  {r['label']:<32}"
        for c in cols:
            v = r.get(c, "—")
            if isinstance(v, float):
                row += f"{v:>10.4f}"
            else:
                row += f"{str(v):>10}"
        print(row)
    print(SEP)


def _delta_table(results: list, baseline_label: str) -> None:
    baseline = next((r for r in results if r["label"] == baseline_label and "error" not in r), None)
    if not baseline:
        return
    cols = ["auroc", "f1", "precision", "recall", "fp"]
    print(f"\n  ΔAUROC vs '{baseline_label}'")
    print("  " + "-" * 60)
    for r in results:
        if "error" in r or r["label"] == baseline_label:
            continue
        parts = [f"  {r['label']:<32}"]
        for c in cols:
            delta = r.get(c, 0) - baseline.get(c, 0)
            sign  = "+" if delta >= 0 else ""
            parts.append(f"  {c}:{sign}{delta:+.4f}")
        print("".join(parts))
    print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_ablation(args) -> list:
    t0 = time.time()
    _banner(f"🔬  ABLATION STUDY — BotOrNot  [{datetime.now():%Y-%m-%d %H:%M}]")

    df = _load_file(args.train)
    _log(f"{len(df):,} lignes × {len(df.columns)} colonnes")

    id_col    = args.id_col    or _find_col(df, ID_PATTERNS)
    label_col = args.label_col or _find_col(df, LABEL_PATTERNS)
    if id_col is None:
        df["_row_id"] = df.index.astype(str); id_col = "_row_id"
    if label_col is None:
        sys.exit("❌ Colonne label non trouvée. Utilisez --label-col.")

    _log(f"ID: {id_col}  |  Label: {label_col}")

    # Résolution modèle
    model_name = args.model
    try:
        import lightgbm; has_lgbm = True
    except ImportError:
        has_lgbm = False
    try:
        import catboost; has_cb = True
    except ImportError:
        has_cb = False
    if model_name == "lgbm"     and not has_lgbm: model_name = "catboost" if has_cb else "lr"
    if model_name == "catboost" and not has_cb:   model_name = "lr"
    _log(f"Modèle : {model_name.upper()}  |  {args.cv_folds}-fold CV")

    _banner("Extraction des features")
    blocks = _extract_all_blocks(df, id_col)

    y_map = df.groupby(id_col)[label_col].max()
    y = y_map.reindex(blocks[id_col]).values.astype(int)
    groups = pd.Series(blocks[id_col]) if pd.Series(blocks[id_col]).nunique() < len(y) else None

    sizes = {k: blocks[k].shape[1] for k in ["tabular","temporal","text_basic","text_model","structural"]}
    _log(f"Blocs : {sizes}")
    _log(f"{len(y):,} comptes  |  bots: {y.mean():.1%}")

    # ── Configurations à tester ─────────────────────────────
    RUNS = [
        # Individuels
        ("tabular_only",         ["tabular"],                            False),
        ("temporal_only",        ["temporal"],                           False),
        ("text_basic_only",      ["text_basic"],                         False),
        ("text_model_only",      ["text_model"],                         False),
        ("structural_only",      ["structural"],                         False),
        # Combinaisons
        ("tab+tmp",              ["tabular", "temporal"],                False),
        ("tab+tmp+txt",          ["tabular", "temporal", "text_basic"],  False),
        ("tab+tmp+txt+tfidf",    ["tabular", "temporal", "text_basic", "text_model"], False),
        # Ensemble
        ("ensemble_tab+tmp+txt", ["tabular", "temporal", "text_basic"],  True),
        ("ensemble_all",         ["tabular","temporal","text_basic","text_model"], True),
        # Ablations (leave-one-out de l'ensemble de base)
        ("ablation_no_tabular",  ["temporal", "text_basic"],             True),
        ("ablation_no_temporal", ["tabular",  "text_basic"],             True),
        ("ablation_no_text",     ["tabular",  "temporal"],               True),
    ]

    _banner(f"Ablation — {len(RUNS)} configurations × {args.cv_folds} folds")
    results = []
    for label, block_names, blend in RUNS:
        t_run = time.time()
        skip_blocks = [b for b in block_names if blocks.get(b, pd.DataFrame()).shape[1] == 0]
        if skip_blocks:
            _log(f"  [{label}]  ⚠️ blocs manquants: {skip_blocks} — SKIPPED")
            results.append({"label": label, "error": f"missing:{skip_blocks}"})
            continue
        _log(f"▶  {label}  (blend={blend})…")
        r = _cv_on_blocks(blocks, id_col, block_names, y, groups,
                          model_name, args.cv_folds, args.seed,
                          label=label, blend=blend)
        elapsed = time.time() - t_run
        if "error" not in r:
            _log(f"   AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}  "
                 f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
                 f"FP={r['fp']}  FN={r['fn']}  ({elapsed:.1f}s)")
        results.append(r)

    _print_table(results)
    _delta_table(results, "ensemble_tab+tmp+txt")

    # Meilleur résultat
    best = max(
        [(r["label"], r["auroc"]) for r in results if "error" not in r],
        key=lambda x: x[1], default=(None, 0)
    )
    _log(f"🏆  Meilleure config : {best[0]}  (AUROC={best[1]:.4f})")

    # Export
    total = time.time() - t0
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    full = {
        "generated_at":   datetime.now().isoformat(),
        "elapsed_seconds": round(total, 1),
        "train": args.train,
        "model": model_name,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "best_config": best[0],
        "best_auroc":  best[1],
        "results": results,
    }
    json_path = args.out + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False, default=str)
    _log(f"JSON → {json_path}")

    csv_path = args.out + ".csv"
    pd.DataFrame([r for r in results if "error" not in r]).to_csv(csv_path, index=False)
    _log(f"CSV  → {csv_path}")

    _banner(f"✅  Ablation terminée en {total:.1f}s")
    return results


def main():
    p = argparse.ArgumentParser(description="🔬 Ablation study BotOrNot")
    p.add_argument("--train",     required=True)
    p.add_argument("--model",     default="lgbm", choices=["lgbm","catboost","lr"])
    p.add_argument("--cv-folds",  type=int, default=5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out",       default="artifacts/ablation")
    p.add_argument("--label-col", default=None)
    p.add_argument("--id-col",    default=None)
    args = p.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
