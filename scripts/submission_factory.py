#!/usr/bin/env python
"""
submission_factory.py — Generateur de soumissions
===================================================
Produit automatiquement 3 fichiers de soumission (conservative,
balanced, aggressive) en un seul run.

Chaque soumission utilise :
  - Le même modèle entraîné une fois sur le dataset complet
  - Les règles anti-FP du profil correspondant
  - Le seuil de décision du profil correspondant
  - Un fichier meta.json complet

Sorties (dans --out) :
  submission_conservative.csv   ← soumission à déposer
  submission_conservative_meta.json
  submission_balanced.csv
  submission_balanced_meta.json
  submission_aggressive.csv
  submission_aggressive_meta.json
  factory_report.json           ← comparaison des 3 profils

Usage :
    python scripts/submission_factory.py --train data/train.csv --test data/test.csv
    python scripts/submission_factory.py --train data/train.csv --test data/test.csv \
        --model catboost --cv-folds 3 --out artifacts/submissions
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from run_baseline import (
    _load_file, _find_col,
    _make_tabular_features, _make_text_features, _make_temporal_features,
    _impute, _get_model, _fit_predict,
    ID_PATTERNS, LABEL_PATTERNS,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.anti_fp import AntiFPFilter, AntiFPConfig

SEP = "─" * 72


def _banner(m): print(f"\n{SEP}\n  {m}\n{SEP}")
def _log(m):    print(f"  [{datetime.now():%H:%M:%S}] {m}")


# ─────────────────────────────────────────────────────────────
# Profils intégrés (miroir de configs/submission_profiles.yaml)
# ─────────────────────────────────────────────────────────────

PROFILES = {
    "conservative": {
        "threshold_mode":     "fixed",
        "threshold_value":    0.60,
        "anti_fp": AntiFPConfig(
            enabled=True,
            min_modules_for_bot=2,
            unilateral_penalty=0.15,
            power_user_protection=True,
            pu_min_followers=3_000,
            pu_follower_penalty=0.08,
            pu_verified_penalty=0.10,
            conflict_rules_enabled=True,
            conflict_penalty=0.10,
        ),
        "calibration": "isotonic",
        "description": "Précision maximale — seuil haut, anti-FP fort",
    },
    "balanced": {
        "threshold_mode":     "f1_optimal",
        "threshold_value":    None,         # calculé sur OOF
        "anti_fp": AntiFPConfig(
            enabled=True,
            min_modules_for_bot=1,
            unilateral_penalty=0.08,
            power_user_protection=True,
            pu_min_followers=10_000,
            pu_follower_penalty=0.04,
            pu_verified_penalty=0.05,
            conflict_rules_enabled=True,
            conflict_penalty=0.06,
        ),
        "calibration": "isotonic",
        "description": "F1 optimal — seuil auto-ajusté, anti-FP modéré",
    },
    "aggressive": {
        "threshold_mode":     "fixed",
        "threshold_value":    0.38,
        "anti_fp": AntiFPConfig(enabled=False),
        "calibration": "sigmoid",
        "description": "Recall maximal — seuil bas, pas d'anti-FP",
    },
}


# ─────────────────────────────────────────────────────────────
# Extraction des features
# ─────────────────────────────────────────────────────────────

def _extract_features(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    tab = _make_tabular_features(df, id_col)
    tab = tab.groupby(id_col).first().reset_index()
    tmp = _make_temporal_features(df, id_col)
    txt = _make_text_features(df, id_col)

    base = tab[[id_col]].copy()
    def _m(block):
        m = base.merge(block, on=id_col, how="left")
        return m.drop(columns=[id_col]).select_dtypes(include=[np.number])

    t = pd.concat([_m(tab), _m(tmp), _m(txt)], axis=1)
    t = t.loc[:, ~t.columns.duplicated()]
    t[id_col] = tab[id_col].values
    return t


def _extract_block_probas(df, id_col, y_train, model_name, n_folds, seed, groups):
    """Calcule les probas OOF pour chaque bloc séparément (pour anti-FP)."""
    block_probas = {}
    blocks = {
        "tabular":   _make_tabular_features(df, id_col).groupby(id_col).first().reset_index(),
        "temporal":  _make_temporal_features(df, id_col),
        "text_basic": _make_text_features(df, id_col),
    }
    base_ids = list(blocks["tabular"][id_col])

    for bname, block_df in blocks.items():
        try:
            feat = block_df.set_index(id_col).reindex(base_ids).reset_index()
            X = _impute(feat.drop(columns=[id_col]).select_dtypes(include=[np.number])).values
            if X.shape[1] == 0:
                continue
            splitter = _get_splitter(X, y_train, groups, n_folds, seed)
            oof = np.zeros(len(y_train))
            for fold, (tr, va) in enumerate(splitter, 1):
                m = _get_model(model_name, seed + fold)
                _, p = _fit_predict(m, X[tr], y_train[tr], X[va])
                oof[va] = p
            block_probas[bname] = oof
        except Exception as e:
            _log(f"  block_probas[{bname}] skipped: {e}")
    return block_probas


def _get_splitter(X, y, groups, n_folds, seed):
    if groups is not None and groups.nunique() >= n_folds * 2:
        return list(GroupKFold(n_splits=n_folds).split(X, y, groups=groups))
    return list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=seed).split(X, y))


# ─────────────────────────────────────────────────────────────
# Seuil optimal
# ─────────────────────────────────────────────────────────────

def _find_f1_threshold(y, proba, lo=0.25, hi=0.80, step=0.01):
    best_t, best_f1 = 0.50, 0.0
    for t in np.arange(lo, hi, step):
        f = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(float(best_t), 3), round(float(best_f1), 4)


# ─────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────

def _compute_metrics(y, proba, threshold, label="") -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "profile":   label,
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
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────
# Générer une soumission pour un profil donné
# ─────────────────────────────────────────────────────────────

def _generate_for_profile(
    profile_name: str,
    profile_cfg: dict,
    oof_proba: np.ndarray,
    test_proba: np.ndarray,
    y_train: np.ndarray,
    train_ids: list,
    test_ids: list,
    feature_df: pd.DataFrame,
    block_probas: dict,
    id_col: str,
    label_col: str,
    out_dir: str,
    meta_base: dict,
    args=None,
) -> dict:
    _log(f"  Profil [{profile_name}] — {profile_cfg['description']}")

    af_filter = AntiFPFilter(profile_cfg["anti_fp"])

    # ── Anti-FP sur les probas OOF (train) ───────────────────
    oof_df = pd.DataFrame({
        id_col: train_ids,
        "proba": oof_proba,
    })
    oof_adjusted = af_filter.apply(oof_df, feature_df, block_probas)
    proba_adj = oof_adjusted["proba_adjusted"].values

    n_triggered = oof_adjusted["anti_fp_triggered"].sum()
    _log(f"    Anti-FP : {n_triggered}/{len(oof_proba)} comptes ajustés")

    # ── Seuil ────────────────────────────────────────────────
    if profile_cfg["threshold_mode"] == "f1_optimal":
        threshold, oof_f1 = _find_f1_threshold(y_train, proba_adj)
        # Garde-fous
        threshold = max(0.45, min(0.65, threshold))
        _log(f"    Seuil F1-optimal : {threshold:.3f} (F1={oof_f1:.4f})")
    else:
        threshold = profile_cfg["threshold_value"]
        oof_f1 = f1_score(y_train, (proba_adj >= threshold).astype(int), zero_division=0)
        _log(f"    Seuil fixe : {threshold:.3f} (F1 OOF={oof_f1:.4f})")

    # ── Métriques OOF ────────────────────────────────────────
    metrics = _compute_metrics(y_train, proba_adj, threshold, label=profile_name)
    _log(f"    AUROC={metrics['auroc']:.4f}  F1={metrics['f1']:.4f}  "
         f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
         f"FP={metrics['fp']}  FN={metrics['fn']}")

    # ── Anti-FP sur le test ───────────────────────────────────
    test_df_pfx = pd.DataFrame({id_col: test_ids, "proba": test_proba})
    # block_probas n'est pas disponible pour le test → None
    test_adj = af_filter.apply(test_df_pfx, feature_df=None)
    test_proba_adj = test_adj["proba_adjusted"].values

    # ── Soumission ────────────────────────────────────────────
    labels = (test_proba_adj >= threshold).astype(int)
    submission = pd.DataFrame({
        id_col:   test_ids,
        "proba":  np.round(test_proba_adj, 6),
        label_col: labels,
    })

    csv_path  = os.path.join(out_dir, f"submission_{profile_name}.csv")
    json_path = os.path.join(out_dir, f"submission_{profile_name}_meta.json")

    submission.to_csv(csv_path, index=False)

    meta = dict(meta_base)
    meta.update({
        "profile": profile_name,
        "description": profile_cfg["description"],
        "threshold": threshold,
        "calibration": profile_cfg["calibration"],
        "anti_fp_enabled": profile_cfg["anti_fp"].enabled,
        "anti_fp_triggered": int(n_triggered),
        "predicted_bots": int(labels.sum()),
        "predicted_humans": int((labels == 0).sum()),
        "bot_rate": round(float(labels.mean()), 4),
        "oof_metrics": metrics,
        "output_csv": csv_path,
    })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    _log(f"    ✅  {csv_path}  ({labels.sum()} bots / {len(labels)} comptes)")

    # ── Format Officiel (.txt) si demandé ────────────────────────────
    official_format = getattr(args, 'format', 'csv') if args else 'csv'
    team_name = getattr(args, 'team_name', 'BotOrNot') if args else 'BotOrNot'
    if official_format == "official":
        bot_ids = [uid for uid, lab in zip(test_ids, labels) if lab == 1]
        txt_path = os.path.join(out_dir, f"{team_name}.detections.{profile_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for uid in bot_ids:
                f.write(f"{uid}\n")
        _log(f"    🏆  {txt_path}  ({len(bot_ids)} bots détectés)")
        meta["output_txt"] = txt_path

    return metrics


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_factory(args) -> None:
    t_start = time.time()
    _banner(f"🏭  SUBMISSION FACTORY — BotOrNot  [{datetime.now():%Y-%m-%d %H:%M}]")

    # ── Chargement ───────────────────────────────────────────
    _log(f"Train : {args.train}")
    df_train = _load_file(args.train)
    _log(f"  {len(df_train):,} lignes × {len(df_train.columns)} colonnes")

    has_test = args.test and os.path.exists(args.test)
    if has_test:
        _log(f"Test  : {args.test}")
        df_test = _load_file(args.test)
        _log(f"  {len(df_test):,} lignes × {len(df_test.columns)} colonnes")
    else:
        _log("⚠️  Pas de fichier test fourni — simulation sur train (80/20)")
        split_idx = int(0.8 * df_train[df_train.columns[0]].nunique())
        all_ids   = df_train[df_train.columns[0]].unique()
        train_ids_set = set(all_ids[:split_idx])
        df_test   = df_train[~df_train[df_train.columns[0]].isin(train_ids_set)]
        df_train  = df_train[df_train[df_train.columns[0]].isin(train_ids_set)]

    id_col    = args.id_col    or _find_col(df_train, ID_PATTERNS) or "user_id"
    label_col = args.label_col or _find_col(df_train, LABEL_PATTERNS)
    if label_col is None:
        sys.exit("❌ Colonne label non trouvée. Utilisez --label-col.")

    # ── Modèle ───────────────────────────────────────────────
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

    # ── Features train ────────────────────────────────────────
    _banner("Extraction des features")
    _log("Train features…")
    feat_train = _extract_features(df_train, id_col)
    y_s = df_train.groupby(id_col)[label_col].max()
    account_order = list(feat_train[id_col])
    y_train = y_s.reindex(account_order).values.astype(int)

    X_all = _impute(feat_train.drop(columns=[id_col]).select_dtypes(include=[np.number]))
    _log(f"  {len(y_train)} comptes  |  features: {X_all.shape[1]}  |  bots: {y_train.mean():.1%}")

    groups_s = pd.Series(account_order)
    groups   = groups_s if groups_s.nunique() < len(y_train) else None

    # ── CV + OOF ─────────────────────────────────────────────
    _banner(f"Cross-validation OOF — {args.cv_folds} folds")
    splitter = _get_splitter(X_all.values, y_train, groups, args.cv_folds, args.seed)
    oof_proba = np.zeros(len(y_train))
    for fold, (tr, va) in enumerate(splitter, 1):
        _log(f"  Fold {fold}/{args.cv_folds}…")
        m = _get_model(model_name, args.seed + fold)
        _, p = _fit_predict(m, X_all.values[tr], y_train[tr], X_all.values[va])
        oof_proba[va] = p
    _log(f"  OOF AUROC : {roc_auc_score(y_train, oof_proba):.4f}")

    # ── Block probas (pour anti-FP) ───────────────────────────
    _log("Calcul block_probas (anti-FP)…")
    block_probas = _extract_block_probas(
        df_train, id_col, y_train, model_name, args.cv_folds, args.seed, groups
    )

    # ── Entraînement complet sur train ────────────────────────
    _banner("Entraînement modèle complet")
    final_model = _get_model(model_name, args.seed)
    final_model.fit(X_all.values, y_train)

    # ── Features test ─────────────────────────────────────────
    _log("Test features…")
    feat_test   = _extract_features(df_test, id_col)
    test_ids    = list(feat_test[id_col])
    X_test      = _impute(feat_test.drop(columns=[id_col]).select_dtypes(include=[np.number]))

    # Aligner les colonnes
    for c in X_all.columns:
        if c not in X_test.columns:
            X_test[c] = np.nan
    X_test = X_test[X_all.columns]
    X_test = _impute(X_test)

    test_proba = final_model.predict_proba(X_test.values)[:, 1]
    _log(f"  {len(test_ids)} comptes test | proba moyenne: {test_proba.mean():.4f}")

    # ── Génération des 3 soumissions ──────────────────────────
    os.makedirs(args.out, exist_ok=True)
    _banner("Génération des soumissions")

    meta_base = {
        "generated_at":    datetime.now().isoformat(),
        "train":           args.train,
        "test":            args.test,
        "model":           model_name,
        "cv_folds":        args.cv_folds,
        "seed":            args.seed,
        "n_train_accounts": len(y_train),
        "n_test_accounts":  len(test_ids),
        "oof_auroc":       round(float(roc_auc_score(y_train, oof_proba)), 4),
        "n_features":      int(X_all.shape[1]),
    }

    all_metrics = []
    for pname, pcfg in PROFILES.items():
        m = _generate_for_profile(
            profile_name=pname,
            profile_cfg=pcfg,
            oof_proba=oof_proba.copy(),
            test_proba=test_proba.copy(),
            y_train=y_train,
            train_ids=account_order,
            test_ids=test_ids,
            feature_df=feat_train.drop(columns=[id_col]) if id_col in feat_train.columns else feat_train,
            block_probas=block_probas,
            id_col=id_col,
            label_col=label_col,
            out_dir=args.out,
            meta_base=meta_base,
            args=args,
        )
        all_metrics.append(m)

    # ── Rapport de comparaison ─────────────────────────────────
    report_path = os.path.join(args.out, "factory_report.json")
    report = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "model": model_name,
        "oof_auroc": meta_base["oof_auroc"],
        "profiles": all_metrics,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # ── Résumé tableau ────────────────────────────────────────
    _banner("📊  COMPARAISON DES PROFILS")
    cols = ["profile", "auroc", "f1", "precision", "recall", "fp", "fn", "threshold"]
    hdr  = f"  {'profil':<18}" + "".join(f"{c:>12}" for c in cols[1:])
    print(hdr)
    print("  " + "-" * (16 + 12 * (len(cols) - 1)))
    for m in all_metrics:
        if "error" in m:
            continue
        row = f"  {m['profile']:<18}"
        for c in cols[1:]:
            v = m.get(c, "—")
            row += f"{v:>12.4f}" if isinstance(v, float) else f"{str(v):>12}"
        print(row)

    total = time.time() - t_start
    _banner(f"✅  Factory terminée en {total:.1f}s")
    _log(f"Rapport global : {report_path}")
    _log(f"Soumissions   : {args.out}/submission_{{conservative,balanced,aggressive}}.csv")


def main():
    p = argparse.ArgumentParser(description="🏭 Submission Factory BotOrNot")
    p.add_argument("--train",     required=True,  help="Fichier d'entraînement")
    p.add_argument("--test",      default=None,   help="Fichier de test (optionnel)")
    p.add_argument("--model",     default="lgbm", choices=["lgbm","catboost","lr"])
    p.add_argument("--cv-folds",  type=int, default=5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out",       default="artifacts/submissions")
    p.add_argument("--label-col", default=None)
    p.add_argument("--id-col",    default=None)
    p.add_argument("--format",    default="csv", choices=["csv", "official"],
                   help="Format de sortie : csv (classique) ou official (fichier .txt, 1 ID bot/ligne)")
    p.add_argument("--team-name", default="BotOrNot",
                   help="Nom d'équipe pour le nommage des fichiers officiels")
    args = p.parse_args()
    run_factory(args)


if __name__ == "__main__":
    main()
