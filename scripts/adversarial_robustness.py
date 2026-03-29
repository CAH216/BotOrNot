#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/adversarial_robustness.py
==================================
Mission 18 — Audit de robustesse adversariale du pipeline.

Crée 3 scénarios synthétiques de bots furtifs qui imitent des humains,
les injecte dans le dataset d'entraînement (en remplacement des bots naïfs),
et mesure la chute de performance du pipeline.

Scénarios :
  1. Sleeper Bot   : bursts + silence 8-12h simulant le sommeil humain
  2. Jitter Bot    : délais aléatoires entre posts (casse la régularité)
  3. LLM Bot       : 2-3 posts/jour, texte fluide, peu de répétition

Ne modifie PAS golden_baseline. C'est un audit pur.
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_baseline import (
    _load_file, _find_col, _impute, _get_model, _fit_predict,
    ID_PATTERNS, LABEL_PATTERNS,
)
from submission_factory import _extract_features

SEP = "=" * 70
RNG = np.random.RandomState(42)

# ─────────────────────────────────────────────────────────────
# Pool de textes fluides pour les bots adversariaux
# ─────────────────────────────────────────────────────────────

HUMAN_TEXTS = [
    "Just had a great coffee at the new place downtown, highly recommend it!",
    "Working from home today, the weather is absolutely gorgeous outside.",
    "Finished reading that book everyone's been talking about. Mixed feelings.",
    "Can't believe it's already March, this year is flying by honestly.",
    "Spent the afternoon organizing my desk and it feels so much better now.",
    "Watched the sunset from the balcony, sometimes the simple things matter most.",
    "Tried making sourdough again, turned out way better than last time!",
    "Long walk in the park with the dog, she found every single puddle.",
    "Traffic was insane this morning. Left 20 minutes early and still late.",
    "Finally got around to updating my portfolio site, small wins count.",
    "Movie night with friends, we picked the worst film but laughed a lot.",
    "Meal prep Sunday is my new favourite habit, saves so much time.",
    "The new season of that show dropped, binge mode activated.",
    "Morning run was rough but the endorphins afterwards are always worth it.",
    "Spent the weekend at a small music festival, discovered some amazing bands.",
    "Trying to learn guitar, my fingers are not happy about it yet.",
    "Got my test results back, everything looks good. Relief flooding in.",
    "Rainy day, perfect excuse to stay in and do absolutely nothing.",
    "Volunteered at the food bank today. Always puts things in perspective.",
    "Planning a trip for next month, can't decide between mountains or coast.",
]


def _compute_metrics(y_true, proba, threshold=0.5):
    """Compute standard metrics."""
    pred = (proba >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"AUROC": 0, "PR-AUC": 0, "F1": 0, "Precision": 0, "Recall": 0, "FP": 0, "FN": 0}
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AUROC": round(float(roc_auc_score(y_true, proba)), 4),
        "PR-AUC": round(float(average_precision_score(y_true, proba)), 4),
        "F1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "Precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "FP": int(fp),
        "FN": int(fn),
    }


# ─────────────────────────────────────────────────────────────
# Générateurs de scénarios adversariaux
# ─────────────────────────────────────────────────────────────

def _get_human_rows(df, id_col, label_col):
    """Extrait les lignes des vrais humains pour servir de modèle."""
    humans = df[df[label_col] == 0]
    return humans


def _gen_sleeper_bot(bot_id, n_posts, base_dt):
    """
    Sleeper Bot : poste en bursts de 3-5 posts espacés de 1-5 min,
    puis dort 8-12h, puis re-burst.  Texte fluide.
    """
    rows = []
    t = base_dt + timedelta(hours=RNG.uniform(0, 6))
    posts_done = 0

    while posts_done < n_posts:
        burst_size = min(RNG.randint(2, 5), n_posts - posts_done)
        for _ in range(burst_size):
            text = RNG.choice(HUMAN_TEXTS)
            row = {}
            row["_uid"] = bot_id
            row["_ts"] = t.isoformat()
            row["_text"] = text
            row["_label"] = 1
            rows.append(row)
            t += timedelta(minutes=RNG.uniform(1, 5))
            posts_done += 1
        # Sleep 8-12h
        t += timedelta(hours=RNG.uniform(8, 12))

    return rows


def _gen_jitter_bot(bot_id, n_posts, base_dt):
    """
    Jitter Bot : délai aléatoire entre chaque post (30s - 2h),
    distribution log-normale pour casser la régularité.  Texte fluide.
    """
    rows = []
    t = base_dt + timedelta(hours=RNG.uniform(0, 12))

    for _ in range(n_posts):
        text = RNG.choice(HUMAN_TEXTS)
        row = {}
        row["_uid"] = bot_id
        row["_ts"] = t.isoformat()
        row["_text"] = text
        row["_label"] = 1
        rows.append(row)
        # Log-normal delay: median ~10 min, sometimes 30s, sometimes 2h
        delay_sec = np.exp(RNG.normal(6.2, 1.0))  # mean ~500s, wide variance
        t += timedelta(seconds=max(30, min(7200, delay_sec)))

    return rows


def _gen_llm_bot(bot_id, n_posts, base_dt):
    """
    LLM Bot : 2-3 posts/jour, spread over multiple days.
    Texte fluide unique (pas de répétition visible).
    """
    rows = []
    t = base_dt
    texts_used = set()

    for i in range(n_posts):
        # Pick a unique text
        available = [t for t in HUMAN_TEXTS if t not in texts_used]
        if not available:
            texts_used.clear()
            available = list(HUMAN_TEXTS)
        text = RNG.choice(available)
        texts_used.add(text)

        # Post during human hours (8am - 11pm)
        post_hour = RNG.uniform(8, 23)
        post_dt = t.replace(hour=int(post_hour), minute=RNG.randint(0, 59))

        row = {}
        row["_uid"] = bot_id
        row["_ts"] = post_dt.isoformat()
        row["_text"] = text
        row["_label"] = 1
        rows.append(row)

        # Every 2-3 posts, advance to next day
        if (i + 1) % RNG.randint(2, 4) == 0:
            t += timedelta(days=1)

    return rows


def _gen_camouflage_bot(bot_id, n_posts, base_dt, human_ipt_stats=None):
    """
    Camouflage Bot : imite PARFAITEMENT un humain.
    - Timing : samplé depuis la distribution réelle des humains (IPT log-normal)
    - Horaires : poste uniquement 8h-23h, surtout 9h-18h
    - Volume : 10 posts répartis sur plusieurs jours
    - Texte : fluide, varié, aucune répétition
    """
    rows = []
    # Human IPT: mean ~3910s, std ~1890s, CV ~0.48
    ipt_mean = human_ipt_stats.get("ipt_mean", 3900) if human_ipt_stats else 3900
    ipt_std = human_ipt_stats.get("ipt_std", 1900) if human_ipt_stats else 1900

    # Start on day 1 at a human hour (8-10 AM)
    t = base_dt + timedelta(hours=RNG.uniform(0, 2))
    texts_used = set()

    for i in range(n_posts):
        # Pick unique text
        available = [tx for tx in HUMAN_TEXTS if tx not in texts_used]
        if not available:
            texts_used.clear()
            available = list(HUMAN_TEXTS)
        text = RNG.choice(available)
        texts_used.add(text)

        # Ensure posting during human hours (8-23h)
        while t.hour < 8 or t.hour >= 23:
            t += timedelta(hours=1)

        row = {}
        row["_uid"] = bot_id
        row["_ts"] = t.isoformat()
        row["_text"] = text
        row["_label"] = 1
        rows.append(row)

        # Sample next delay from human-like distribution (log-normal fitting)
        # Log-normal params derived from human IPT: mu=8.15, sigma=0.46
        delay = np.exp(RNG.normal(np.log(ipt_mean) - 0.5 * (ipt_std / ipt_mean) ** 2,
                                   ipt_std / ipt_mean))
        delay = max(600, min(14400, delay))  # clamp 10 min - 4 hours
        t += timedelta(seconds=delay)

        # Cross midnight? Skip to next morning
        if t.hour < 8:
            t = t.replace(hour=8, minute=RNG.randint(0, 59))

    return rows


def build_adversarial_dataset(df_original, id_col, label_col, scenario_name, generator_fn):
    """
    Replace the bot rows in the original dataset with adversarially-crafted ones.
    Humans stay untouched.
    """
    humans_df = df_original[df_original[label_col] == 0].copy()
    bot_ids = df_original[df_original[label_col] == 1][id_col].unique()

    # Build per-bot randomized profiles drawn from human distribution
    human_stats = humans_df.groupby(id_col).first()
    profile_cols = [c for c in ["followers_count", "following_count", "statuses_count"] if c in human_stats.columns]
    has_verified = "verified" in human_stats.columns

    ts_col = _find_col(df_original, ["created_at", "timestamp", "date"])
    text_col = _find_col(df_original, ["text", "content", "tweet"])
    base_dt = datetime(2023, 1, 1, 8, 0, 0)

    # Generate adversarial bots
    adv_rows = []
    n_posts_per_bot = 10  # match original

    for bot_id in bot_ids:
        if generator_fn == _gen_camouflage_bot:
            # Pass human IPT stats to camouflage generator
            human_ts = pd.to_datetime(humans_df[ts_col], utc=True, errors='coerce') if ts_col else None
            ipt_stats = {}
            if human_ts is not None:
                ipt_all = human_ts.groupby(humans_df[id_col]).apply(
                    lambda x: x.sort_values().diff().dt.total_seconds().dropna()
                )
                if len(ipt_all) > 0:
                    ipt_stats = {"ipt_mean": float(ipt_all.mean()), "ipt_std": float(ipt_all.std())}
            rows = generator_fn(bot_id, n_posts_per_bot, base_dt, human_ipt_stats=ipt_stats)
        else:
            rows = generator_fn(bot_id, n_posts_per_bot, base_dt)
        adv_rows.extend(rows)

    # Build DataFrame
    adv_records = []
    for r in adv_rows:
        rec = {
            id_col: r["_uid"],
            label_col: r["_label"],
        }
        if ts_col:
            rec[ts_col] = r["_ts"]
        if text_col:
            rec[text_col] = r["_text"]
        # Draw random profile from human distribution for each row
        if profile_cols:
            random_human = human_stats.iloc[RNG.randint(0, len(human_stats))]
            for k in profile_cols:
                rec[k] = random_human[k]
        if has_verified:
            rec["verified"] = False
        adv_records.append(rec)

    adv_df = pd.DataFrame(adv_records)

    # Combine humans + adversarial bots
    combined = pd.concat([humans_df, adv_df], ignore_index=True)
    return combined


def run_cv_eval(df, id_col, label_col, n_folds=5, seed=42):
    """Run cross-validated LightGBM and return OOF predictions + feature importances."""
    feat_df = _extract_features(df, id_col)

    y_series = df.groupby(id_col)[label_col].max()
    account_order = list(feat_df[id_col])
    y = y_series.reindex(account_order).fillna(0).values.astype(int)

    # CRITICAL: exclude label column from features to prevent data leakage
    leak_patterns = [label_col, f"raw_{label_col}"]
    feature_cols = [c for c in feat_df.columns if c != id_col and c not in leak_patterns]
    X = _impute(feat_df[feature_cols].select_dtypes(include=[np.number])).values
    used_cols = feat_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

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

    # Map importances to feature families
    family_imp = {"temporal": 0, "tabular": 0, "text": 0, "structural": 0, "other": 0}
    for col, imp in zip(used_cols, importances):
        if col.startswith("tmp_"):
            family_imp["temporal"] += imp
        elif col.startswith("txt_"):
            family_imp["text"] += imp
        elif col.startswith("tab_") or col.startswith("raw_") or col.startswith("bool_"):
            family_imp["tabular"] += imp
        elif col.startswith("struct_"):
            family_imp["structural"] += imp
        else:
            family_imp["other"] += imp

    # Top individual features
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [(used_cols[i], float(importances[i])) for i in top_idx]

    return y, oof, family_imp, top_features


def main():
    print(f"\n{SEP}")
    print(" 🛡️  MISSION 18 — AUDIT DE ROBUSTESSE ADVERSARIALE")
    print(SEP)

    # Load baseline dataset
    train_path = "data/_dryrun_train.csv"
    if not os.path.exists(train_path):
        sys.exit(f"❌ Fichier introuvable : {train_path}")

    df = _load_file(train_path)
    id_col = _find_col(df, ID_PATTERNS) or "user_id"
    label_col = _find_col(df, LABEL_PATTERNS)
    if not label_col:
        sys.exit("❌ Colonne label introuvable.")

    scenarios = OrderedDict([
        ("baseline", None),
        ("sleeper_bot", _gen_sleeper_bot),
        ("jitter_bot", _gen_jitter_bot),
        ("llm_bot", _gen_llm_bot),
        ("camouflage_bot", _gen_camouflage_bot),
    ])

    all_results = {}
    baseline_metrics = None

    for name, gen_fn in scenarios.items():
        print(f"\n{'─'*50}")
        print(f"  📊 Scénario : {name.upper()}")
        print(f"{'─'*50}")

        if gen_fn is None:
            eval_df = df
        else:
            print(f"  Génération du dataset adversarial ({name})...")
            eval_df = build_adversarial_dataset(df, id_col, label_col, name, gen_fn)

        n_bots = eval_df[eval_df[label_col] == 1][id_col].nunique()
        n_humans = eval_df[eval_df[label_col] == 0][id_col].nunique()
        print(f"  Comptes: {n_bots} bots + {n_humans} humains = {n_bots + n_humans} total")

        y, oof, family_imp, top_feats = run_cv_eval(eval_df, id_col, label_col)
        metrics = _compute_metrics(y, oof)

        print(f"  AUROC={metrics['AUROC']}  PR-AUC={metrics['PR-AUC']}  "
              f"F1={metrics['F1']}  Prec={metrics['Precision']}  Rec={metrics['Recall']}  "
              f"FP={metrics['FP']}  FN={metrics['FN']}")

        if name == "baseline":
            baseline_metrics = metrics
            delta_str = "(référence)"
        else:
            d_auroc = metrics["AUROC"] - baseline_metrics["AUROC"]
            d_f1 = metrics["F1"] - baseline_metrics["F1"]
            delta_str = f"ΔAUROC={d_auroc:+.4f}  ΔF1={d_f1:+.4f}"
        print(f"  {delta_str}")

        # Risk assessment
        if name != "baseline":
            f1_drop = baseline_metrics["F1"] - metrics["F1"]
            if f1_drop > 0.20:
                risk = "🔴 HIGH-RISK"
            elif f1_drop > 0.08:
                risk = "🟡 WATCH"
            else:
                risk = "🟢 SAFE"
            print(f"  Évaluation du risque : {risk} (chute F1 = {f1_drop:+.4f})")

        all_results[name] = {
            "metrics": metrics,
            "family_importance": family_imp,
            "top_features": top_feats,
        }

    # ─────────────────────────────────────────────────────────
    # Rapport Final
    # ─────────────────────────────────────────────────────────
    out_dir = Path("artifacts/adversarial")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON export
    json_path = out_dir / "adversarial_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Markdown report
    md_path = out_dir / "adversarial_robustness_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Rapport de Robustesse Adversariale\n\n")
        f.write(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Comparison table
        f.write("## 1. Comparaison des Performances\n\n")
        f.write("| Scénario | AUROC | PR-AUC | F1-Score | Precision | Recall | FP | FN | ΔAUROC | ΔF1 | Risque |\n")
        f.write("|----------|-------|--------|----------|-----------|--------|----|----|--------|-----|--------|\n")

        bl = all_results["baseline"]["metrics"]
        for name, data in all_results.items():
            m = data["metrics"]
            if name == "baseline":
                da, df1, risk = "—", "—", "Réf."
            else:
                da = f"{m['AUROC'] - bl['AUROC']:+.4f}"
                df1_val = bl["F1"] - m["F1"]
                df1 = f"{m['F1'] - bl['F1']:+.4f}"
                risk = "🔴 HIGH" if df1_val > 0.20 else "🟡 WATCH" if df1_val > 0.08 else "🟢 SAFE"

            f.write(f"| **{name}** | {m['AUROC']} | {m['PR-AUC']} | {m['F1']} | "
                    f"{m['Precision']} | {m['Recall']} | {m['FP']} | {m['FN']} | "
                    f"{da} | {df1} | {risk} |\n")

        # Feature family analysis
        f.write("\n## 2. Importance des Familles de Features par Scénario\n\n")
        f.write("| Scénario | Temporal | Tabular | Text | Structural | Autre |\n")
        f.write("|----------|----------|---------|------|------------|-------|\n")

        for name, data in all_results.items():
            fi = data["family_importance"]
            total = sum(fi.values()) or 1
            f.write(f"| **{name}** | {fi['temporal']/total*100:.1f}% | {fi['tabular']/total*100:.1f}% | "
                    f"{fi['text']/total*100:.1f}% | {fi['structural']/total*100:.1f}% | {fi['other']/total*100:.1f}% |\n")

        # Top features baseline
        f.write("\n## 3. Top 10 Features (Baseline)\n\n")
        f.write("| Rang | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for i, (feat, imp) in enumerate(all_results["baseline"]["top_features"], 1):
            f.write(f"| {i} | `{feat}` | {imp:.1f} |\n")

        # Vulnerability analysis
        f.write("\n## 4. Analyse des Vulnérabilités\n\n")

        # Find worst scenario
        worst_name = None
        worst_drop = 0
        for name in ["sleeper_bot", "jitter_bot", "llm_bot", "camouflage_bot"]:
            drop = bl["F1"] - all_results[name]["metrics"]["F1"]
            if drop > worst_drop:
                worst_drop = drop
                worst_name = name

        f.write(f"### Menace la plus dangereuse\n")
        if worst_name:
            f.write(f"- **{worst_name}** avec une chute de F1 de **{worst_drop:+.4f}**\n\n")
        else:
            f.write("- Aucune menace significative détectée.\n\n")

        f.write("### Modules qui résistent le mieux\n")
        bl_fi = all_results["baseline"]["family_importance"]
        for name in ["sleeper_bot", "jitter_bot", "llm_bot", "camouflage_bot"]:
            fi = all_results[name]["family_importance"]
            f.write(f"- **{name}** : ")
            diffs = []
            for fam in ["temporal", "tabular", "text"]:
                bl_pct = bl_fi[fam] / max(sum(bl_fi.values()), 1) * 100
                sc_pct = fi[fam] / max(sum(fi.values()), 1) * 100
                delta = sc_pct - bl_pct
                if abs(delta) > 2:
                    diffs.append(f"{fam} {'↑' if delta > 0 else '↓'}{abs(delta):.1f}%")
            if diffs:
                f.write(", ".join(diffs) + "\n")
            else:
                f.write("pas de changement significatif d'importance\n")

        f.write("\n### Recommandation Globale\n\n")
        max_drop = max(
            bl["F1"] - all_results[s]["metrics"]["F1"]
            for s in ["sleeper_bot", "jitter_bot", "llm_bot", "camouflage_bot"]
        )
        if max_drop > 0.20:
            f.write("> ⚠️ **Niveau : HIGH-RISK** — Au moins un scénario adversarial provoque une chute massive.\n")
            f.write("> Proposer des fixes derrière flag (ex: features anti-jitter, analyse de diversité textuelle).\n")
        elif max_drop > 0.08:
            f.write("> 🟡 **Niveau : WATCH** — Dégradation modérée détectée.\n")
            f.write("> Le pipeline reste fonctionnel mais des améliorations ciblées sont recommandées.\n")
        else:
            f.write("> 🟢 **Niveau : SAFE** — Le pipeline résiste bien aux scénarios adversariaux testés.\n")
            f.write("> Aucune action immédiate nécessaire.\n")

    print(f"\n{SEP}")
    print(f"  🎯 Audit terminé !")
    print(f"  📄 Rapport : {md_path}")
    print(f"  📦 Données : {json_path}")
    print(SEP)


if __name__ == "__main__":
    main()
