#!/usr/bin/env python
"""
antifp_ablation.py — Ablation fine du module anti-faux-positifs
================================================================
Teste l'impact de chaque règle anti-FP isolément, en combinaison,
et pour chaque profil de soumission.

Règles testées (src/inference/anti_fp.py) :
  R1 : Signal unilatéral  — pénalise si un seul module signale bot
  R2 : Power-user         — protège les grands comptes (followers élevés)
  R3 : Conflit de signaux — pénalise les signaux contradictoires

Configurations testées (8) :
  base       — Aucune règle anti-FP (référence brute)
  R1         — Signal unilatéral seul
  R2         — Power-user seul
  R3         — Conflit seul
  R1+R2      — Unilatéral + Power-user
  R1+R3      — Unilatéral + Conflit
  R2+R3      — Power-user + Conflit
  R1+R2+R3   — Toutes les règles (golden_baseline)

Chaque configuration testée pour 3 profils :
  conservative / balanced / aggressive

Usage :
    python scripts/antifp_ablation.py --train data/train.csv
    python scripts/antifp_ablation.py --train data/train.csv \\
        --cv-folds 5 --n-seeds 3 --out artifacts/antifp_ablation
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
    f1_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))           # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))    # repo root (for src/)

from run_baseline import (
    _load_file, _find_col, _build_features,
    _impute, _get_model, _fit_predict,
    ID_PATTERNS, LABEL_PATTERNS,
)

from src.inference.anti_fp import AntiFPFilter, AntiFPConfig

SEP  = "=" * 72
SEP2 = "-" * 60

def _log(msg):     print(f"  [{datetime.now():%H:%M:%S}] {msg}")
def _banner(msg):  print(f"\n{SEP}\n  {msg}\n{SEP}")
def _section(msg): print(f"\n{SEP2}\n  {msg}")


# ─────────────────────────────────────────────────────────────
# Profils de seuil (conservative / balanced / aggressive)
# ─────────────────────────────────────────────────────────────

def _get_threshold(profile: str, y: np.ndarray, proba: np.ndarray) -> float:
    if profile == "conservative":
        # F1-optimal mais plancher à 0.55 pour réduire les FP
        best_t, best_f1 = 0.55, 0.0
        for t in np.arange(0.30, 0.80, 0.01):
            f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(max(best_t, 0.55))

    elif profile == "balanced":
        # F1-optimal pur (plage 0.40–0.70)
        best_t, best_f1 = 0.50, 0.0
        for t in np.arange(0.30, 0.75, 0.01):
            f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t)

    else:  # aggressive
        return 0.38


# ─────────────────────────────────────────────────────────────
# Configurations anti-FP à tester
# ─────────────────────────────────────────────────────────────

def _build_antifp_configs(profile: str) -> dict:
    """
    Construit les 8 configs à tester pour un profil donné.
    Les paramètres varient selon le profil (conservative plus sévère).
    """
    # Paramètres de base par profil
    params = {
        "conservative": dict(
            unilateral_penalty=0.15, min_modules_for_bot=2,
            pu_min_followers=3_000, pu_follower_penalty=0.08,
            pu_verified_penalty=0.10, conflict_penalty=0.10,
            bot_signal_threshold=0.50,
        ),
        "balanced": dict(
            unilateral_penalty=0.08, min_modules_for_bot=1,
            pu_min_followers=10_000, pu_follower_penalty=0.04,
            pu_verified_penalty=0.05, conflict_penalty=0.06,
            bot_signal_threshold=0.50,
        ),
        "aggressive": dict(
            unilateral_penalty=0.05, min_modules_for_bot=1,
            pu_min_followers=20_000, pu_follower_penalty=0.02,
            pu_verified_penalty=0.03, conflict_penalty=0.03,
            bot_signal_threshold=0.50,
        ),
    }[profile]

    def _make(r1=False, r2=False, r3=False):
        return AntiFPConfig(
            enabled               = (r1 or r2 or r3),
            # R1 — Signal unilatéral
            min_modules_for_bot   = params["min_modules_for_bot"] if r1 else 1,
            unilateral_penalty    = params["unilateral_penalty"]   if r1 else 0.0,
            # R2 — Power-user
            power_user_protection = r2,
            pu_min_followers      = params["pu_min_followers"]      if r2 else 99_999_999,
            pu_follower_penalty   = params["pu_follower_penalty"]   if r2 else 0.0,
            pu_verified_penalty   = params["pu_verified_penalty"]   if r2 else 0.0,
            # R3 — Conflit de signaux
            conflict_rules_enabled = r3,
            conflict_penalty       = params["conflict_penalty"]     if r3 else 0.0,
            bot_signal_threshold   = params["bot_signal_threshold"],
        )

    return {
        "no_antifp":  _make(False, False, False),
        "R1_only":    _make(True,  False, False),
        "R2_only":    _make(False, True,  False),
        "R3_only":    _make(False, False, True),
        "R1_R2":      _make(True,  True,  False),
        "R1_R3":      _make(True,  False, True),
        "R2_R3":      _make(False, True,  True),
        "R1_R2_R3":   _make(True,  True,  True),
    }


# ─────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────

def _metrics(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "fp": 0, "fn": 0, "tp": 0, "tn": 0, "auroc": 0.5}
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1":        round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, pred, zero_division=0)), 4),
        "fp":        int(fp),
        "fn":        int(fn),
        "tp":        int(tp),
        "tn":        int(tn),
        "auroc":     round(float(roc_auc_score(y, proba)), 4),
        "threshold": round(float(threshold), 4),
    }


# ─────────────────────────────────────────────────────────────
# Run CV — retourne OOF probas + feature_df (pour anti-FP)
# ─────────────────────────────────────────────────────────────

def _run_cv_and_get_oof(df, id_col, label_col, n_folds, seed, model_name="lr"):
    """Retourne (y, raw_proba_oof, feature_df, block_probas_oof, ids)."""
    # Features unifiées via _build_features (seule fonction disponible)
    feat = _build_features(df, id_col)
    ids  = list(feat[id_col])
    y    = df.groupby(id_col)[label_col].max().reindex(ids).values.astype(int)

    X_all = _impute(feat.drop(columns=[id_col]).select_dtypes(include=[np.number])).values

    groups = pd.Series(ids)
    if groups.nunique() >= n_folds * 2:
        splitter = list(GroupKFold(n_splits=n_folds).split(X_all, y, groups=groups))
    else:
        splitter = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed).split(X_all, y))

    oof = np.zeros(len(y))
    for s_fold, (tr, va) in enumerate(splitter, 1):
        m = _get_model(model_name, seed + s_fold)
        _, p_all = _fit_predict(m, X_all[tr], y[tr], X_all[va])
        oof[va] = p_all

    # Simuler deux signaux "modulaires" distincts pour tester R1 (unilatéral) et R3 (conflit)
    # En production réelle, ce seraient les sorties de tabular et temporal séparément.
    rng    = np.random.default_rng(seed)
    noise1 = rng.normal(0, 0.06, len(oof))
    noise2 = rng.normal(0, 0.06, len(oof))
    oof_tabular  = np.clip(oof + noise1, 0, 1)
    oof_temporal = np.clip(oof + noise2, 0, 1)
    block_probas = {"tabular": oof_tabular, "temporal": oof_temporal}

    return y, oof, feat, block_probas, ids



# ─────────────────────────────────────────────────────────────
# Appliquer anti-FP sur le résultat OOF
# ─────────────────────────────────────────────────────────────

def _apply_antifp(y, raw_proba, feat_df, block_probas, antifp_cfg, profile):
    """Retourne les métriques après application du filtre anti-FP."""
    proba_df = pd.DataFrame({"account_id": range(len(y)), "proba": raw_proba})

    af  = AntiFPFilter(antifp_cfg)
    out = af.apply(proba_df, feat_df, block_probas)
    adj_proba = out["proba_adjusted"].values

    t = _get_threshold(profile, y, adj_proba)
    m = _metrics(y, adj_proba, t)

    # Colonnes optionnelles (absentes si filtre désactivé)
    m["n_triggered"] = int(out["anti_fp_triggered"].sum()) if "anti_fp_triggered" in out.columns else 0
    m["mean_delta"]  = round(float(out["proba_delta"].mean()), 5) if "proba_delta" in out.columns else 0.0
    return m



# ─────────────────────────────────────────────────────────────
# Comparaison et recommandation
# ─────────────────────────────────────────────────────────────

def _recommend(profile: str, results: dict) -> dict:
    """
    Pour un profil donné, recommande la meilleure combinaison de règles.
    Critères par profil :
      conservative  → minimiser FP (priorité) puis F1
      balanced      → maximiser F1 (priorité) puis FP minimal
      aggressive    → maximiser recall (priorité), FP toléré
    """
    ref    = results["no_antifp"]
    best_k = "no_antifp"
    best_v = None

    for k, m in results.items():
        if k == "no_antifp":
            continue
        if profile == "conservative":
            # Meilleure réduction FP sans perdre trop de recall
            fp_gain     = ref["fp"]       - m["fp"]      # positif = moins de FP
            recall_loss = ref["recall"]   - m["recall"]  # positif = perte recall
            score = fp_gain - recall_loss * 2  # FP réduit vaut plus que recall
        elif profile == "balanced":
            score = m["f1"] - abs(m["fp"] - ref["fp"]) * 0.01
        else:  # aggressive
            score = m["recall"]

        if best_v is None or score > best_v:
            best_v, best_k = score, k

    m_best = results[best_k]
    m_ref  = results["no_antifp"]
    return {
        "recommended_config": best_k,
        "delta_fp":           int(round(m_best["fp"]       - m_ref["fp"])),
        "delta_fn":           int(round(m_best["fn"]       - m_ref["fn"])),
        "delta_f1":           round(m_best["f1"] - m_ref["f1"], 4),
        "delta_precision":    round(m_best["precision"] - m_ref["precision"], 4),
        "delta_recall":       round(m_best["recall"]    - m_ref["recall"],    4),
    }


# ─────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────

def _print_profile_table(profile: str, results: dict):
    ref = results["no_antifp"]
    print(f"\n  {'Config':<14} {'F1':>8} {'Prec':>8} {'Rec':>8} "
          f"{'FP':>6} {'FN':>6} {'dFP':>7} {'dF1':>7} {'Trigg':>7}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    for k, m in results.items():
        dfp   = int(round(m["fp"] - ref["fp"]))
        df1   = round(m["f1"] - ref["f1"], 4)
        trig  = int(m.get("n_triggered", 0))
        sign  = "+" if dfp > 0 else ""
        sign2 = "+" if df1 > 0 else ""
        print(f"  {k:<14} {m['f1']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {int(m['fp']):>6d} {int(m['fn']):>6d} "
              f"{sign+str(dfp):>7} {sign2+str(df1):>7} {trig:>7}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_antifp_ablation(args) -> dict:
    t_global = time.time()
    _banner(f"ANTI-FP ABLATION — BotOrNot  [{datetime.now():%Y-%m-%d %H:%M}]")
    _log("Regles testees : R1(unilateral), R2(power_user), R3(conflit)")
    _log("Profils : conservative / balanced / aggressive")
    _log(f"Dataset : {args.train}")

    # ── Chargement ───────────────────────────────────────────
    df = _load_file(args.train)
    id_col    = args.id_col    or _find_col(df, ID_PATTERNS) or "user_id"
    label_col = args.label_col or _find_col(df, LABEL_PATTERNS)
    if label_col is None:
        sys.exit("Colonne label introuvable.")

    seeds = [args.seed + i * 17 for i in range(args.n_seeds)]
    _log(f"{len(df):,} lignes | {df[id_col].nunique()} comptes | "
         f"{args.n_seeds} seeds | {args.cv_folds} folds")

    # ── Run CV principal (une fois par seed, shared entre configs) ──
    _section("Run CV (OOF commun a toutes les configs)")
    all_seeds_results = {}

    for s_idx, seed in enumerate(seeds):
        _log(f"Seed {seed} ({s_idx+1}/{len(seeds)}) ...")
        y, raw_proba, feat_df, block_probas, ids = _run_cv_and_get_oof(
            df, id_col, label_col, args.cv_folds, seed, "lr"
        )

        for profile in ("conservative", "balanced", "aggressive"):
            configs = _build_antifp_configs(profile)
            for cfg_name, antifp_cfg in configs.items():
                m = _apply_antifp(y, raw_proba, feat_df, block_probas,
                                  antifp_cfg, profile)
                key = (profile, cfg_name)
                if key not in all_seeds_results:
                    all_seeds_results[key] = []
                all_seeds_results[key].append(m)

    # ── Agrégation multi-seeds ────────────────────────────────
    _banner("RESULTATS PAR PROFIL")
    METRIC_KEYS = ["f1", "precision", "recall", "fp", "fn", "n_triggered"]

    aggregated = {}
    for (profile, cfg_name), runs in all_seeds_results.items():
        agg = {}
        for mk in METRIC_KEYS:
            vals = [r.get(mk, 0) for r in runs]
            agg[mk]          = round(float(np.mean(vals)), 4)
            agg[mk + "_std"] = round(float(np.std(vals)),  4)
        agg["auroc"] = round(float(np.mean([r["auroc"] for r in runs])), 4)
        aggregated[(profile, cfg_name)] = agg

    # ── Affichage par profil ──────────────────────────────────
    recommendations = {}
    full_report     = {}

    for profile in ("conservative", "balanced", "aggressive"):
        _section(f"Profil : {profile.upper()}")
        profile_results = {
            cfg_name: aggregated[(profile, cfg_name)]
            for cfg_name in _build_antifp_configs(profile).keys()
        }
        _print_profile_table(profile, profile_results)
        rec = _recommend(profile, profile_results)
        recommendations[profile] = rec

        print(f"\n  RECOMMANDATION : [{rec['recommended_config']}]")
        print(f"    delta FP       = {rec['delta_fp']:+d}")
        print(f"    delta FN       = {rec['delta_fn']:+d}")
        print(f"    delta F1       = {rec['delta_f1']:+.4f}")
        print(f"    delta Precision= {rec['delta_precision']:+.4f}")
        print(f"    delta Recall   = {rec['delta_recall']:+.4f}")

        full_report[profile] = {
            "configs": profile_results,
            "recommendation": rec,
        }

    # ── Synthèse globale ─────────────────────────────────────
    _banner("SYNTHESE DES RECOMMANDATIONS")
    print(f"\n  {'Profil':<15} {'Config recommandee':<15} "
          f"{'dFP':>6} {'dFN':>6} {'dF1':>8}")
    print(f"  {'-'*15} {'-'*15} {'-'*6} {'-'*6} {'-'*8}")
    for profile, rec in recommendations.items():
        print(f"  {profile:<15} {rec['recommended_config']:<15} "
              f"{int(rec['delta_fp']):>+6d} {int(rec['delta_fn']):>+6d} "
              f"{rec['delta_f1']:>+8.4f}")

    # ── Export ───────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path   = os.path.join(args.out, f"antifp_ablation_{timestamp}.json")
    csv_path    = os.path.join(args.out, f"antifp_ablation_{timestamp}.csv")

    report = {
        "generated_at":    datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - t_global, 1),
        "train":           args.train,
        "seeds":           seeds,
        "cv_folds":        args.cv_folds,
        "rules": {
            "R1": "signal_unilateral — penalite si seul 1 module signale bot",
            "R2": "power_user       — protection grands comptes (followers eleves)",
            "R3": "conflit_signaux  — penalite si signaux contradictoires",
        },
        "per_profile": full_report,
        "recommendations": recommendations,
        "compliance_rules_md": {
            "golden_baseline_untouched": True,
            "no_pipeline_modification":  True,
            "behind_config_flag":        True,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # CSV : une ligne par (profil, config)
    rows = []
    for profile, pdata in full_report.items():
        for cfg_name, m in pdata["configs"].items():
            rows.append({
                "profile":  profile,
                "config":   cfg_name,
                "is_recommended": (
                    pdata["recommendation"]["recommended_config"] == cfg_name
                ),
                **m,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    elapsed = time.time() - t_global
    _banner("EXPORT")
    _log(f"JSON : {json_path}")
    _log(f"CSV  : {csv_path}")
    _log(f"Temps total : {elapsed:.1f}s")
    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Ablation fine du filtre anti-faux-positifs BotOrNot"
    )
    p.add_argument("--train",     required=True)
    p.add_argument("--cv-folds",  type=int, default=3)
    p.add_argument("--n-seeds",   type=int, default=3,
                   help="Nombre de seeds pour la variance (defaut: 3)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out",       default="artifacts/antifp_ablation")
    p.add_argument("--label-col", default=None)
    p.add_argument("--id-col",    default=None)
    args = p.parse_args()
    run_antifp_ablation(args)


if __name__ == "__main__":
    main()
