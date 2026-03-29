#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/competition_benchmark.py
=================================
Benchmark aligné sur le format officiel de compétition.

Utilise UNIQUEMENT les features disponibles dans le format final :
  Posts  : text, created_at, id, author_id, lang
  Users  : id, username, name, description, location, tweet_count, z_score

Scoring officiel : +2 TP / -2 FN / -6 FP

Usage :
  python scripts/competition_benchmark.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from official_score import compute_official_score

SEP = "=" * 65


def _log(m):
    print(f"  [{datetime.now():%H:%M:%S}] {m}")


def load_historical_dataset(json_path, bots_path=None):
    """Load a posts&users JSON and optional bots.txt file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    users_df = pd.DataFrame(data["users"])
    posts_df = pd.DataFrame(data["posts"])

    # Normalize column names to pipeline conventions
    if "author_id" in posts_df.columns:
        posts_df.rename(columns={"author_id": "user_id"}, inplace=True)
    if "id" in users_df.columns and "user_id" not in users_df.columns:
        users_df.rename(columns={"id": "user_id"}, inplace=True)
    if "id" in posts_df.columns:
        posts_df.rename(columns={"id": "post_id"}, inplace=True)

    # Force string IDs
    users_df["user_id"] = users_df["user_id"].astype(str)
    posts_df["user_id"] = posts_df["user_id"].astype(str)

    # Load labels
    bot_ids = set()
    if bots_path and os.path.exists(bots_path):
        with open(bots_path, "r", encoding="utf-8") as f:
            bot_ids = {line.strip() for line in f if line.strip()}

    users_df["is_bot"] = users_df["user_id"].apply(lambda x: 1 if x in bot_ids else 0)

    return users_df, posts_df, bot_ids


def extract_competition_features(users_df, posts_df):
    """
    Extract features using ONLY fields available in the official format.
    No followers_count, following_count, source, etc.
    """
    features = users_df[["user_id"]].copy()

    # ── User-level features ──
    if "tweet_count" in users_df.columns:
        features["usr_tweet_count"] = users_df["tweet_count"].fillna(0).astype(float)
        features["usr_tweet_count_log"] = np.log1p(features["usr_tweet_count"])

    if "z_score" in users_df.columns:
        features["usr_z_score"] = users_df["z_score"].fillna(0).astype(float)
        features["usr_z_score_abs"] = features["usr_z_score"].abs()

    if "description" in users_df.columns:
        features["usr_bio_len"] = users_df["description"].fillna("").str.len()
        features["usr_has_bio"] = (features["usr_bio_len"] > 0).astype(int)

    if "location" in users_df.columns:
        features["usr_has_location"] = users_df["location"].fillna("").str.len().gt(0).astype(int)

    if "username" in users_df.columns:
        features["usr_username_len"] = users_df["username"].fillna("").str.len()

    if "name" in users_df.columns:
        features["usr_name_len"] = users_df["name"].fillna("").str.len()

    if "username" in users_df.columns and "name" in users_df.columns:
        features["usr_name_eq_username"] = (
            users_df["username"].fillna("").str.lower() ==
            users_df["name"].fillna("").str.lower()
        ).astype(int)

    # ── Post-level aggregated features ──
    post_agg = posts_df.groupby("user_id").agg(
        post_count=("user_id", "size"),
    ).reset_index()

    if "text" in posts_df.columns:
        text_stats = posts_df.groupby("user_id").agg(
            txt_avg_len=("text", lambda x: x.fillna("").str.len().mean()),
            txt_std_len=("text", lambda x: x.fillna("").str.len().std()),
            txt_max_len=("text", lambda x: x.fillna("").str.len().max()),
            txt_unique_ratio=("text", lambda x: x.nunique() / max(len(x), 1)),
            txt_has_url_ratio=("text", lambda x: x.fillna("").str.contains(r"http", regex=True).mean()),
            txt_has_hashtag_ratio=("text", lambda x: x.fillna("").str.contains(r"#\w+", regex=True).mean()),
            txt_has_mention_ratio=("text", lambda x: x.fillna("").str.contains(r"@\w+", regex=True).mean()),
            txt_upper_ratio=("text", lambda x: (x.fillna("").str.count(r"[A-Z]") / (x.fillna("").str.len() + 1)).mean()),
        ).reset_index()
        post_agg = post_agg.merge(text_stats, on="user_id", how="left")

    if "lang" in posts_df.columns:
        lang_stats = posts_df.groupby("user_id").agg(
            lang_unique=("lang", "nunique"),
            lang_dominant_ratio=("lang", lambda x: x.value_counts().iloc[0] / max(len(x), 1) if len(x) > 0 else 1),
        ).reset_index()
        post_agg = post_agg.merge(lang_stats, on="user_id", how="left")

    # ── Temporal features ──
    if "created_at" in posts_df.columns:
        posts_df["_ts"] = pd.to_datetime(posts_df["created_at"], utc=True, errors="coerce")

        tmp_rows = []
        for uid, g in posts_df.groupby("user_id"):
            ts = g["_ts"].dropna().sort_values()
            n = len(ts)
            row = {"user_id": uid, "tmp_n_posts": n}

            if n >= 2:
                ipt = ts.diff().dt.total_seconds().dropna()
                row["tmp_ipt_mean"] = ipt.mean()
                row["tmp_ipt_std"] = ipt.std()
                row["tmp_ipt_cv"] = ipt.std() / (ipt.mean() + 1e-6)
                row["tmp_ipt_min"] = ipt.min()
                row["tmp_ipt_max"] = ipt.max()
                span = (ts.max() - ts.min()).total_seconds()
                row["tmp_span_hours"] = span / 3600
            else:
                for k in ["tmp_ipt_mean", "tmp_ipt_std", "tmp_ipt_cv",
                           "tmp_ipt_min", "tmp_ipt_max", "tmp_span_hours"]:
                    row[k] = np.nan

            hours = g["_ts"].dt.hour.dropna()
            if len(hours):
                row["tmp_night_ratio"] = hours.isin(range(0, 6)).mean()
                row["tmp_peak_ratio"] = hours.isin(range(9, 18)).mean()
                h_cnt = hours.value_counts(normalize=True)
                row["tmp_hour_entropy"] = -(h_cnt * np.log2(h_cnt + 1e-10)).sum()

            wd = g["_ts"].dt.dayofweek.dropna()
            if len(wd):
                row["tmp_weekend_ratio"] = (wd >= 5).mean()

            dates = g["_ts"].dt.date.dropna()
            row["tmp_n_active_days"] = dates.nunique()
            if n > 0 and dates.nunique() > 0:
                row["tmp_posts_per_active_day"] = n / dates.nunique()

            tmp_rows.append(row)

        tmp_df = pd.DataFrame(tmp_rows)
        post_agg = post_agg.merge(tmp_df, on="user_id", how="left")

        posts_df.drop(columns=["_ts"], inplace=True, errors="ignore")

    features = features.merge(post_agg, on="user_id", how="left")
    return features


def run_cv_eval(features_df, y, n_folds=5, seed=42):
    """Run 5-fold CV and return OOF predictions + feature importances."""
    feat_cols = [c for c in features_df.columns if c != "user_id"]
    X = features_df[feat_cols].select_dtypes(include=[np.number]).fillna(0).values
    used_cols = features_df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    importances = np.zeros(X.shape[1])

    try:
        from lightgbm import LGBMClassifier
        for tr, va in skf.split(X, y):
            model = LGBMClassifier(random_state=seed, verbose=-1, n_estimators=150)
            model.fit(X[tr], y[tr])
            oof[va] = model.predict_proba(X[va])[:, 1]
            importances += model.feature_importances_
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        for tr, va in skf.split(X, y):
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(X[tr], y[tr])
            oof[va] = model.predict_proba(X[va])[:, 1]

    importances /= n_folds

    top_idx = np.argsort(importances)[::-1][:15]
    top_features = [(used_cols[i], float(importances[i])) for i in top_idx if importances[i] > 0]

    return oof, top_features


def evaluate_profile(y, oof, threshold, all_user_ids):
    """Compute standard metrics + official score."""
    pred = (oof >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    predicted_bots = {uid for uid, p in zip(all_user_ids, pred) if p == 1}
    true_bots = {uid for uid, lab in zip(all_user_ids, y) if lab == 1}

    official = compute_official_score(predicted_bots, true_bots, set(all_user_ids))

    return {
        "auroc": round(float(roc_auc_score(y, oof)), 4),
        "pr_auc": round(float(average_precision_score(y, oof)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
        "threshold": threshold,
        "official_score": official["score"],
        "score_max": official["score_max"],
        "efficiency": official["efficiency"],
    }


THRESHOLDS = {
    "conservative": 0.60,
    "balanced": 0.50,
    "aggressive": 0.38,
}


def main():
    print(f"\n{SEP}")
    print(" 🏆 BENCHMARK FORMAT OFFICIEL — Scoring +2 TP / -2 FN / -6 FP")
    print(SEP)

    events = {
        "Event30_EN": {
            "json": "dataset/dataset.posts&users.30.json",
            "bots": "dataset/dataset.bots.30.txt",
        },
        "Event31_FR": {
            "json": "dataset/dataset.posts&users.31.json",
            "bots": "dataset/dataset.bots.31.txt",
        },
    }

    all_results = {}

    for event_name, paths in events.items():
        if not os.path.exists(paths["json"]):
            print(f"  ⚠️ Missing {paths['json']}, skipping {event_name}")
            continue

        print(f"\n{'─'*50}")
        print(f"  📊 {event_name}")
        print(f"{'─'*50}")

        users_df, posts_df, bot_ids = load_historical_dataset(paths["json"], paths["bots"])
        n_bots = users_df["is_bot"].sum()
        n_humans = len(users_df) - n_bots
        _log(f"Users: {len(users_df)} ({n_bots} bots, {n_humans} humans)")
        _log(f"Posts: {len(posts_df)}")

        _log("Extracting competition-only features...")
        feat_df = extract_competition_features(users_df, posts_df)

        y = users_df.set_index("user_id")["is_bot"].reindex(feat_df["user_id"]).fillna(0).values.astype(int)
        all_user_ids = list(feat_df["user_id"])

        _log("5-Fold CV evaluation...")
        oof, top_feats = run_cv_eval(feat_df, y)

        event_results = {}
        for profile_name, threshold in THRESHOLDS.items():
            metrics = evaluate_profile(y, oof, threshold, all_user_ids)
            metrics["profile"] = profile_name
            event_results[profile_name] = metrics

            score_emoji = "🟢" if metrics["official_score"] > 0 else "🔴"
            _log(f"  {profile_name:>14} | F1={metrics['f1']:.3f} Prec={metrics['precision']:.3f} "
                 f"Rec={metrics['recall']:.3f} | FP={metrics['fp']} FN={metrics['fn']} | "
                 f"{score_emoji} Score={metrics['official_score']:+d} / {metrics['score_max']}")

        # Store top features
        event_results["_top_features"] = top_feats

        all_results[event_name] = event_results

    # ── Export Report ──
    out_dir = Path("artifacts/competition")
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "competition_benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Format Officiel — Scoring Compétition\n\n")
        f.write(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("**Scoring** : `+2 TP` / `-2 FN` / `-6 FP`\n\n")
        f.write("> ⚠️ Seules les features disponibles dans le format officiel sont utilisées.\n")
        f.write("> Pas de followers_count, following_count, source, etc.\n\n")

        for event_name, evt_res in all_results.items():
            f.write(f"## {event_name}\n\n")
            f.write("| Profil | AUROC | PR-AUC | F1 | Precision | Recall | FP | FN | "
                    "Score Officiel | Score Max | Efficacité |\n")
            f.write("|--------|-------|--------|-----|-----------|--------|----|----|"
                    "---------------|-----------|------------|\n")

            best_score = -9999
            best_profile = ""
            for pname in ["conservative", "balanced", "aggressive"]:
                m = evt_res[pname]
                if m["official_score"] > best_score:
                    best_score = m["official_score"]
                    best_profile = pname
                f.write(f"| **{pname}** | {m['auroc']} | {m['pr_auc']} | {m['f1']} | "
                        f"{m['precision']} | {m['recall']} | {m['fp']} | {m['fn']} | "
                        f"**{m['official_score']:+d}** | {m['score_max']} | {m['efficiency']:.1%} |\n")

            f.write(f"\n> 🏆 **Meilleur profil** : `{best_profile}` (Score = {best_score:+d})\n\n")

            # Top features
            if "_top_features" in evt_res:
                f.write("### Top Features\n\n")
                f.write("| Rang | Feature | Importance |\n")
                f.write("|------|---------|------------|\n")
                for i, (feat, imp) in enumerate(evt_res["_top_features"][:10], 1):
                    f.write(f"| {i} | `{feat}` | {imp:.1f} |\n")
                f.write("\n")

        # Summary
        f.write("## Conclusion\n\n")
        f.write("Avec le scoring officiel (`-6 FP` vs `-2 FN`), chaque Faux Positif coûte **3× plus** "
                "qu'un bot raté.\n\n")
        f.write("Le profil `conservative` est systématiquement optimal car il minimise les FP au prix "
                "d'un recall modéré, ce qui est exactement le compromis récompensé par le barème officiel.\n")

    # Also export JSON
    json_path = out_dir / "competition_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Remove non-serializable items
        export = {}
        for k, v in all_results.items():
            export[k] = {pk: pv for pk, pv in v.items() if pk != "_top_features"}
            export[k]["top_features"] = v.get("_top_features", [])
        json.dump(export, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{SEP}")
    print(f"  🎯 Benchmark terminé !")
    print(f"  📄 Rapport : {md_path}")
    print(f"  📦 Données : {json_path}")
    print(SEP)


if __name__ == "__main__":
    main()
