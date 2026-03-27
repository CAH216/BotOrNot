#!/usr/bin/env python
"""
run_baseline.py — Mode baseline d'urgence (pipeline compétitif en < 5 minutes)
================================================================================
Composition :
  • Ingestion + normalisation
  • Features tabular + temporal (Plan A ou B selon données)
  • LightGBM (ou CatBoost en fallback, ou LR en dernier recours)
  • CV group-k-fold (ou stratified si pas d'account_id)
  • Seuil conservateur (par défaut 0.5, ajustable)
  • Export soumission CSV

Usage :
    python scripts/run_baseline.py --train data/train.csv --test data/test.csv
    python scripts/run_baseline.py --train data/train.csv --test data/test.csv \\
        --model catboost --threshold 0.55 --out artifacts/submissions/baseline

Options :
    --train       Chemin fichier d'entraînement (obligatoire)
    --test        Chemin fichier de test (optionnel, pour export)
    --model       lgbm | catboost | lr  (défaut: lgbm)
    --cv-folds    Nombre de folds CV (défaut: 5)
    --threshold   Seuil de décision (défaut: 0.5, conservateur recommandé: 0.55)
    --seed        Seed global (défaut: 42)
    --out         Préfixe de sortie (défaut: artifacts/submissions/baseline)
    --label-col   Nom de la colonne label dans le train (défaut: auto-détecté)
    --id-col      Nom colonne ID (défaut: auto-détecté)
    --no-cv       Désactive la CV (entraîne sur tout le train directement)
"""
import sys
import os
import json
import time
import warnings
import argparse
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Imports obligatoires ─────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("❌ numpy et pandas requis : pip install numpy pandas")

# ─── Imports scikit-learn ─────────────────────────────────────────────────
try:
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
except ImportError:
    sys.exit("❌ scikit-learn requis : pip install scikit-learn")

# ─── Imports optionnels ───────────────────────────────────────────────────
HAS_LGBM = False
HAS_CB   = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    pass

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────
# Constantes — patterns de colonnes
# ──────────────────────────────────────────────────────────────────────────

ID_PATTERNS    = ["user_id", "account_id", "author_id", "userid", "uid"]
LABEL_PATTERNS = ["label", "is_bot", "bot", "class", "target", "y"]
TS_PATTERNS    = ["created_at", "timestamp", "date", "post_time", "posted_at", "datetime"]
TEXT_PATTERNS  = ["text", "content", "tweet", "post", "body", "message"]

SEP = "─" * 60


# ──────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────

def _banner(msg: str):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print('─'*60)


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


def _find_col(df, patterns):
    low = {c.lower(): c for c in df.columns}
    for p in patterns:
        if p in low:
            return low[p]
    return None


def _load_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"❌ Fichier introuvable : {path}")
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    if ext in (".json", ".jsonl"):
        try:
            return pd.read_json(path, lines=(ext == ".jsonl"))
        except Exception:
            return pd.read_json(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    sys.exit(f"❌ Format non supporté : {ext}")


# ──────────────────────────────────────────────────────────────────────────
# Feature engineering minimal
# ──────────────────────────────────────────────────────────────────────────

def _make_tabular_features(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Features tabulaires de base à partir des colonnes numériques et ratios."""
    feat = pd.DataFrame({id_col: df[id_col]})

    # Colonnes numériques brutes
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if c == id_col:
            continue
        feat[f"raw_{c}"] = df[c]

    # Ratios classiques
    followers_col = _find_col(df, ["followers_count", "followers", "follower_count"])
    following_col = _find_col(df, ["following_count", "following", "friends_count"])
    statuses_col  = _find_col(df, ["statuses_count", "status_count", "tweet_count"])
    listed_col    = _find_col(df, ["listed_count", "listed"])

    if followers_col and following_col:
        f = df[followers_col].clip(lower=0) + 1
        fg = df[following_col].clip(lower=0) + 1
        feat["tab_ff_ratio"]         = (f / fg).clip(upper=1000)
        feat["tab_followers_log"]    = np.log1p(df[followers_col].clip(lower=0))
        feat["tab_following_log"]    = np.log1p(df[following_col].clip(lower=0))
        feat["tab_ff_sum"]           = df[followers_col].clip(lower=0) + df[following_col].clip(lower=0)
        feat["tab_ff_asymmetry"]     = (f - fg).abs() / (f + fg)

    if statuses_col:
        feat["tab_statuses_log"]     = np.log1p(df[statuses_col].clip(lower=0))
        if followers_col:
            feat["tab_engagement"]   = (df[followers_col] + 1) / (df[statuses_col] + 1)

    if listed_col:
        feat["tab_listed_log"]       = np.log1p(df[listed_col].clip(lower=0))

    # Colonnes booléennes / catégorielles simples
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        feat[f"bool_{c}"] = df[c].astype(int)

    # Indicateurs de nullité pour colonnes importantes
    int_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    high_miss = [c for c in int_cols if df[c].isna().mean() > 0.1]
    for c in high_miss:
        feat[f"miss_{c}"] = df[c].isna().astype(int)

    return feat


def _make_text_features(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Features textuelles légères par compte."""
    text_col = _find_col(df, TEXT_PATTERNS)
    if text_col is None:
        return pd.DataFrame({id_col: df[id_col].unique()})

    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    t = df[text_col]

    df["txt_len"]           = t.str.len()
    df["txt_word_count"]    = t.str.split().str.len().fillna(0)
    df["txt_has_url"]       = t.str.contains(r"http[s]?://", regex=True).astype(int)
    df["txt_url_count"]     = t.str.count(r"http")
    df["txt_mention_count"] = t.str.count(r"@\w+")
    df["txt_hashtag_count"] = t.str.count(r"#\w+")
    df["txt_excl_count"]    = t.str.count(r"!")
    df["txt_upper_ratio"]   = t.str.count(r"[A-Z]") / (t.str.len() + 1)
    df["txt_digit_ratio"]   = t.str.count(r"\d") / (t.str.len() + 1)
    df["txt_is_empty"]      = (t == "").astype(int)

    txt_cols = [c for c in df.columns if c.startswith("txt_")]
    rows = []
    for uid, g in df[[id_col] + txt_cols].groupby(id_col):
        row = {id_col: uid}
        for c in txt_cols:
            vals = g[c].dropna()
            if len(vals):
                row[f"{c}_mean"] = vals.mean()
                row[f"{c}_std"]  = vals.std() if len(vals) > 1 else 0.0
                row[f"{c}_max"]  = vals.max()
        rows.append(row)
    return pd.DataFrame(rows)




def _make_temporal_features(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Features temporelles par compte (Plan A: précision secondes, Plan B: dates)."""
    ts_col = _find_col(df, TS_PATTERNS)
    if ts_col is None:
        return pd.DataFrame({id_col: df[id_col].unique()})

    df = df.copy()
    try:
        df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    except Exception:
        return pd.DataFrame({id_col: df[id_col].unique()})

    valid = df["_ts"].notna().mean()
    if valid < 0.1:
        _log(f"  ⚠️  Timestamps quasi-vides ({valid:.0%} valides) → plan temporel dégradé")

    df["_hour"]    = df["_ts"].dt.hour
    df["_weekday"] = df["_ts"].dt.dayofweek
    df["_date"]    = df["_ts"].dt.date

    rows = []
    for uid, g in df.groupby(id_col):
        ts = g["_ts"].dropna().sort_values()
        n  = len(ts)

        row = {id_col: uid}
        row["tmp_n_posts"] = n

        if n >= 2:
            ipt = ts.diff().dt.total_seconds().dropna()
            row["tmp_ipt_mean"] = ipt.mean()
            row["tmp_ipt_std"]  = ipt.std()
            row["tmp_ipt_cv"]   = ipt.std() / (ipt.mean() + 1e-6)
            row["tmp_ipt_min"]  = ipt.min()
            row["tmp_ipt_max"]  = ipt.max()
            span = (ts.max() - ts.min()).total_seconds()
            row["tmp_span_hours"] = span / 3600
        else:
            row.update({k: np.nan for k in
                        ["tmp_ipt_mean","tmp_ipt_std","tmp_ipt_cv","tmp_ipt_min","tmp_ipt_max","tmp_span_hours"]})

        # Heures
        hours = g["_hour"].dropna()
        if len(hours):
            row["tmp_night_ratio"] = (hours.isin(range(0, 6))).mean()
            row["tmp_peak_ratio"]  = (hours.isin(range(9, 18))).mean()
            # Entropie des heures
            h_cnt = hours.value_counts(normalize=True)
            row["tmp_hour_entropy"] = -(h_cnt * np.log2(h_cnt + 1e-10)).sum()

        # Jours de la semaine
        wd = g["_weekday"].dropna()
        if len(wd):
            row["tmp_weekend_ratio"] = (wd >= 5).mean()

        # Jours actifs
        dates = g["_date"].dropna()
        row["tmp_n_active_days"] = dates.nunique()
        if n > 0 and dates.nunique() > 0:
            row["tmp_posts_per_active_day"] = n / dates.nunique()

        rows.append(row)

    return pd.DataFrame(rows)


def _build_features(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Construit et fusionne toutes les features."""
    _log("Building tabular features…")
    tab  = _make_tabular_features(df, id_col)

    _log("Building text features…")
    txt  = _make_text_features(df, id_col)

    _log("Building temporal features…")
    tmp  = _make_temporal_features(df, id_col)

    # Pivot sur id_col
    base = tab.groupby(id_col).first().reset_index()
    if len(txt) > 1:
        base = base.merge(txt, on=id_col, how="left")
    if len(tmp) > 1:
        base = base.merge(tmp, on=id_col, how="left")

    # Drop colonnes avec trop de manquants ou variance nulle
    feat_cols = [c for c in base.columns if c != id_col]
    high_miss = [c for c in feat_cols if base[c].isna().mean() > 0.95]
    base = base.drop(columns=high_miss)

    _log(f"Feature matrix : {len(base)} lignes × {len(base.columns)-1} features")
    return base


# ──────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────

def _get_model(model_name: str, seed: int):
    """Retourne le modèle choisi avec paramètres compétitifs et optimaux."""
    if model_name == "lgbm" and HAS_LGBM:
        return lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )

    if model_name == "catboost" and HAS_CB:
        return cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=0,
        )

    # Fallback LR robuste
    _log("⚠️  LightGBM/CatBoost non disponible — fallback LogisticRegression")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )),
    ])


def _fit_predict(model, X_tr, y_tr, X_va):
    """Fit + predict_proba, compatible Pipeline et modèles natifs."""
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_va)[:, 1]
    else:
        proba = model.decision_function(X_va)
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
    return model, proba


def _impute(X: pd.DataFrame) -> pd.DataFrame:
    """Imputation médiane simple pour LightGBM/CatBoost."""
    return X.fillna(X.median(numeric_only=True))


# ──────────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────────

def _run_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
            model_name: str, n_folds: int, seed: int) -> dict:
    """Exécute k-fold CV et retourne les OOF probas + métriques."""

    feature_cols = X.columns.tolist()

    if groups is not None and groups.nunique() >= n_folds * 2:
        _log(f"Stratégie CV : GroupKFold({n_folds}) sur account_id")
        kf = GroupKFold(n_splits=n_folds)
        splitter = kf.split(X, y, groups=groups)
    else:
        _log(f"Stratégie CV : StratifiedKFold({n_folds})")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splitter = kf.split(X, y)

    oof_proba  = np.zeros(len(X))
    fold_aucs  = []
    models     = []

    X_arr = _impute(X).values
    y_arr = y.values

    for fold, (tr_idx, va_idx) in enumerate(splitter, 1):
        t_fold = time.time()
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        model = _get_model(model_name, seed + fold)
        model, proba = _fit_predict(model, X_tr, y_tr, X_va)

        oof_proba[va_idx] = proba
        auc = roc_auc_score(y_va, proba)
        fold_aucs.append(auc)
        models.append(model)

        elapsed = time.time() - t_fold
        _log(f"  Fold {fold}/{n_folds}  →  AUC = {auc:.4f}  ({elapsed:.1f}s)")

    oof_auc = roc_auc_score(y_arr, oof_proba)
    _log(f"  OOF AUC = {oof_auc:.4f}  (std folds = {np.std(fold_aucs):.4f})")

    return {
        "models":     models,
        "oof_proba":  oof_proba,
        "oof_auc":    oof_auc,
        "fold_aucs":  fold_aucs,
        "feature_cols": feature_cols,
    }


# ──────────────────────────────────────────────────────────────────────────
# Seuil optimisé
# ──────────────────────────────────────────────────────────────────────────

def _find_best_threshold(y_true: np.ndarray, proba: np.ndarray,
                         conservative: bool = True) -> dict:
    """Cherche le seuil F1-max, avec option seuil conservateur."""
    best_f1, best_thresh = 0.0, 0.5
    thresholds = np.arange(0.3, 0.8, 0.01)

    for t in thresholds:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t

    # Seuil conservateur = max(f1_optimal, 0.5)
    conservative_thresh = max(best_thresh, 0.5) if conservative else best_thresh

    return {
        "f1_optimal_thresh":     round(float(best_thresh), 4),
        "f1_optimal_score":      round(float(best_f1), 4),
        "conservative_thresh":   round(float(conservative_thresh), 4),
    }


# ──────────────────────────────────────────────────────────────────────────
# Inférence sur test
# ──────────────────────────────────────────────────────────────────────────

def _predict_test(cv_result: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Moyenne des probas sur tous les folds (blending simple)."""
    X_imp = _impute(X_test).values
    probas = []
    for model in cv_result["models"]:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_imp)[:, 1]
        else:
            d = model.decision_function(X_imp)
            p = (d - d.min()) / (d.max() - d.min() + 1e-9)
        probas.append(p)
    return np.mean(probas, axis=0)


# ──────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────

def _export(out_prefix: str, submission_df: pd.DataFrame,
            meta: dict, cv_result: dict) -> None:
    os.makedirs(os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else ".", exist_ok=True)

    # Soumission
    sub_path = out_prefix + ".csv"
    submission_df.to_csv(sub_path, index=False)
    _log(f"Soumission exportée → {sub_path}")

    # Métadonnées
    meta_path = out_prefix + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    _log(f"Métadonnées exportées → {meta_path}")

    # Feature importances (LightGBM / CatBoost)
    imp_path = out_prefix + "_feature_importances.csv"
    try:
        feat_cols = cv_result["feature_cols"]
        importances = []
        for m in cv_result["models"]:
            if hasattr(m, "feature_importances_"):
                importances.append(m.feature_importances_)
        if importances:
            mean_imp = np.mean(importances, axis=0)
            imp_df = pd.DataFrame({
                "feature":    feat_cols,
                "importance": mean_imp,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(imp_path, index=False)
            _log(f"Feature importances → {imp_path}")
            _log("  Top 10 features :")
            for _, row in imp_df.head(10).iterrows():
                _log(f"    {row['feature']:<45} {row['importance']:.1f}")
    except Exception as e:
        _log(f"  (feature importances indisponibles : {e})")


# ──────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────

def run_baseline(args) -> None:
    t_start = time.time()
    _banner(f"🚀  BASELINE D'URGENCE — BotOrNot  [{datetime.now():%Y-%m-%d %H:%M}]")

    # Seed global
    np.random.seed(args.seed)

    # ── 1. Chargement ──────────────────────────────────────────────────────
    _banner("1/5  Chargement des données")
    _log(f"Train : {args.train}")
    df_train = _load_file(args.train)
    _log(f"  {len(df_train):,} lignes × {len(df_train.columns)} colonnes")

    # Détection des colonnes clés
    id_col    = args.id_col    or _find_col(df_train, ID_PATTERNS)
    label_col = args.label_col or _find_col(df_train, LABEL_PATTERNS)

    if id_col is None:
        _log("⚠️  Pas d'account_id détecté → index utilisé comme ID")
        df_train["_row_id"] = df_train.index.astype(str)
        id_col = "_row_id"

    if label_col is None:
        sys.exit("❌ Aucune colonne label détectée. Utilisez --label-col <nom>.")

    _log(f"  ID column    : {id_col}")
    _log(f"  Label column : {label_col}  →  distribution: "
         f"{dict(df_train[label_col].value_counts().head(5))}")

    # ── 2. Feature Engineering ─────────────────────────────────────────────
    _banner("2/5  Feature Engineering")
    X_all = _build_features(df_train, id_col)

    # Aligner les labels sur les accounts (agrégation max = bot si un seul post est bot)
    label_per_account = (df_train.groupby(id_col)[label_col]
                                 .max()
                                 .reset_index()
                                 .rename(columns={label_col: "__label__"}))
    X_all = X_all.merge(label_per_account, on=id_col, how="left")

    y = X_all.pop("__label__").astype(int)
    groups = X_all[id_col] if X_all[id_col].nunique() < len(X_all) else None
    X_ids  = X_all[[id_col]].copy()
    X_feat = X_all.drop(columns=[id_col]).select_dtypes(include=[np.number])

    _log(f"  X shape     : {X_feat.shape}")
    _log(f"  Targets     : {y.value_counts().to_dict()}")
    _log(f"  Label positif (bot) : {y.mean():.1%}")

    # Résolution du modèle
    model_name = args.model
    if model_name == "lgbm" and not HAS_LGBM:
        _log("⚠️  LightGBM non installé → fallback CatBoost")
        model_name = "catboost" if HAS_CB else "lr"
    if model_name == "catboost" and not HAS_CB:
        _log("⚠️  CatBoost non installé → fallback LogisticRegression")
        model_name = "lr"

    _log(f"  Modèle sélectionné : {model_name.upper()}")

    # ── 3. Cross-validation ────────────────────────────────────────────────
    _banner(f"3/5  Cross-Validation ({args.cv_folds}-fold)")
    if args.no_cv:
        _log("⚠️  CV désactivée (--no-cv). Entraînement sur tout le train.")
        model = _get_model(model_name, args.seed)
        X_imp = _impute(X_feat).values
        model.fit(X_imp, y.values)
        cv_result = {
            "models": [model],
            "oof_proba": model.predict_proba(X_imp)[:, 1],
            "oof_auc": roc_auc_score(y.values, model.predict_proba(X_imp)[:,1]),
            "fold_aucs": [],
            "feature_cols": X_feat.columns.tolist(),
        }
    else:
        cv_result = _run_cv(X_feat, y, groups, model_name, args.cv_folds, args.seed)

    # ── 4. Seuil ───────────────────────────────────────────────────────────
    _banner("4/5  Optimisation du seuil")
    thresh_info = _find_best_threshold(y.values, cv_result["oof_proba"], conservative=True)
    final_threshold = args.threshold if args.threshold else thresh_info["conservative_thresh"]

    _log(f"  Seuil F1-optimal    : {thresh_info['f1_optimal_thresh']} "
         f"(F1={thresh_info['f1_optimal_score']:.4f})")
    _log(f"  Seuil conservateur  : {thresh_info['conservative_thresh']}")
    _log(f"  Seuil utilisé       : {final_threshold}")

    oof_pred = (cv_result["oof_proba"] >= final_threshold).astype(int)
    oof_f1  = f1_score(y.values, oof_pred, zero_division=0)
    oof_prec = precision_score(y.values, oof_pred, zero_division=0)
    oof_rec  = recall_score(y.values, oof_pred, zero_division=0)

    _log(f"  OOF AUC              : {cv_result['oof_auc']:.4f}")
    _log(f"  OOF F1 @ {final_threshold}      : {oof_f1:.4f}")
    _log(f"  OOF Precision        : {oof_prec:.4f}")
    _log(f"  OOF Recall           : {oof_rec:.4f}")

    # ── 5. Export ──────────────────────────────────────────────────────────
    _banner("5/5  Export")

    # OOF submission (sur le train)
    oof_sub = X_ids.copy()
    oof_sub["prob_bot"] = cv_result["oof_proba"].round(6)
    oof_sub["label"]    = oof_pred
    oof_sub["label_text"] = oof_sub["label"].map({1: "bot", 0: "human"})

    # Test submission
    if args.test:
        _log(f"Test : {args.test}")
        df_test = _load_file(args.test)
        if id_col not in df_test.columns:
            _log("⚠️  Colonne ID absente dans le test — index utilisé.")
            df_test[id_col] = df_test.index.astype(str)
        X_test_all = _build_features(df_test, id_col)
        X_test_ids = X_test_all[[id_col]].copy()
        X_test_feat = X_test_all.drop(columns=[id_col]).select_dtypes(include=[np.number])
        # Aligner les colonnes
        for c in X_feat.columns:
            if c not in X_test_feat.columns:
                X_test_feat[c] = np.nan
        X_test_feat = X_test_feat[X_feat.columns]
        test_proba  = _predict_test(cv_result, X_test_feat)
        test_pred   = (test_proba >= final_threshold).astype(int)

        test_sub = X_test_ids.copy()
        test_sub["prob_bot"]   = test_proba.round(6)
        test_sub["label"]      = test_pred
        test_sub["label_text"] = test_sub["label"].map({1: "bot", 0: "human"})
        submission_df = test_sub
        _log(f"  Prédictions test : {test_pred.sum():,} bots / {len(test_pred):,} comptes "
             f"({test_pred.mean():.1%})")
    else:
        submission_df = oof_sub
        _log("ℹ️  Pas de fichier test fourni — soumission OOF exportée.")

    elapsed = time.time() - t_start
    meta = {
        "generated_at":       datetime.now().isoformat(),
        "elapsed_seconds":    round(elapsed, 1),
        "train_path":         str(args.train),
        "test_path":          str(args.test) if args.test else None,
        "model":              model_name,
        "n_features":         len(X_feat.columns),
        "n_train_accounts":   len(X_feat),
        "cv_folds":           args.cv_folds,
        "seed":               args.seed,
        "threshold":          final_threshold,
        "oof_auc":            round(cv_result["oof_auc"], 4),
        "oof_f1":             round(float(oof_f1), 4),
        "oof_precision":      round(float(oof_prec), 4),
        "oof_recall":         round(float(oof_rec), 4),
        "fold_aucs":          [round(a, 4) for a in cv_result["fold_aucs"]],
        "threshold_analysis": thresh_info,
    }

    _export(args.out, submission_df, meta, cv_result)

    _banner(f"✅  PIPELINE TERMINÉ en {elapsed:.1f}s")
    print(f"\n  Résumé :")
    print(f"  • Modèle        : {model_name.upper()}")
    print(f"  • OOF AUC       : {cv_result['oof_auc']:.4f}")
    print(f"  • OOF F1        : {oof_f1:.4f} @ seuil {final_threshold}")
    print(f"  • Soumission    : {args.out}.csv")
    print(f"  • Métadonnées   : {args.out}_meta.json")
    print()


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Baseline d'urgence BotOrNot",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train",     required=True,  help="Fichier d'entraînement (csv/json/parquet)")
    parser.add_argument("--test",      default=None,   help="Fichier de test (optionnel)")
    parser.add_argument("--model",     default="lgbm",
                        choices=["lgbm", "catboost", "lr"],
                        help="Modèle : lgbm | catboost | lr (défaut: lgbm)")
    parser.add_argument("--cv-folds",  type=int, default=5, help="Nombre de folds CV (défaut: 5)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Seuil de décision (défaut: auto-optimisé, conservateur)")
    parser.add_argument("--seed",      type=int, default=42, help="Seed global (défaut: 42)")
    parser.add_argument("--out",       default="artifacts/submissions/baseline",
                        help="Préfixe de sortie (défaut: artifacts/submissions/baseline)")
    parser.add_argument("--label-col", default=None, help="Nom de la colonne label")
    parser.add_argument("--id-col",    default=None, help="Nom de la colonne ID compte")
    parser.add_argument("--no-cv",     action="store_true", help="Désactive la CV")
    args = parser.parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
