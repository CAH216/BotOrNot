#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/dual_profile_benchmark.py
===================================
Compare unified conservative profile vs dual EN/FR optimized profiles.

Rule: no change becomes active without measured gain at official scoring.
"""
import json, numpy as np, warnings, sys, os
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from competition_benchmark import load_historical_dataset, extract_competition_features, run_cv_eval
from official_score import compute_official_score

PROFILES = {
    "conservative_unified": {"threshold": 0.60, "label": "Conservative (t=0.60)"},
    "competition_en":       {"threshold": 0.30, "label": "EN-Optimized (t=0.30)"},
    "competition_fr":       {"threshold": 0.60, "label": "FR-Optimized (t=0.60)"},
    "balanced_050":         {"threshold": 0.50, "label": "Balanced (t=0.50)"},
    "aggressive_038":       {"threshold": 0.38, "label": "Aggressive (t=0.38)"},
}

EVENTS = {
    "Event30_EN": {
        "json": "dataset/dataset.posts&users.30.json",
        "bots": "dataset/dataset.bots.30.txt",
        "recommended": "competition_en",
    },
    "Event31_FR": {
        "json": "dataset/dataset.posts&users.31.json",
        "bots": "dataset/dataset.bots.31.txt",
        "recommended": "competition_fr",
    },
}


def evaluate(y, oof, uids, threshold):
    pred_bots = {uid for uid, p in zip(uids, oof) if p >= threshold}
    true_bots = {uid for uid, lab in zip(uids, y) if lab == 1}
    r = compute_official_score(pred_bots, true_bots, set(uids))
    return r


def main():
    print("=" * 65)
    print(" DUAL PROFILE BENCHMARK — EN vs FR")
    print(" Scoring: +2 TP / -2 FN / -6 FP")
    print("=" * 65)

    all_results = {}

    for ev_name, ev_cfg in EVENTS.items():
        if not os.path.exists(ev_cfg["json"]):
            print(f"  Skipping {ev_name} (file not found)")
            continue

        users_df, posts_df, bot_ids = load_historical_dataset(ev_cfg["json"], ev_cfg["bots"])
        feat_df = extract_competition_features(users_df, posts_df)
        y = users_df.set_index("user_id")["is_bot"].reindex(feat_df["user_id"]).fillna(0).values.astype(int)
        uids = list(feat_df["user_id"])
        oof, _ = run_cv_eval(feat_df, y)

        print(f"\n{'─'*50}")
        print(f"  {ev_name}  (bots={y.sum()}, humans={len(y)-y.sum()}, max={2*y.sum()})")
        print(f"  Recommended: {ev_cfg['recommended']}")
        print(f"{'─'*50}")

        ev_results = {}
        for pname, pcfg in PROFILES.items():
            r = evaluate(y, oof, uids, pcfg["threshold"])
            is_recommended = (pname == ev_cfg["recommended"])
            marker = " ◄ RECOMMENDED" if is_recommended else ""
            is_best = False
            ev_results[pname] = {
                "label": pcfg["label"],
                "threshold": pcfg["threshold"],
                "score": r["score"],
                "tp": r["tp"],
                "fp": r["fp"],
                "fn": r["fn"],
                "precision": r["precision"],
                "recall": r["recall"],
                "recommended": is_recommended,
            }
            print(f"  {pcfg['label']:30s} | Score={r['score']:+4d} | TP={r['tp']:3d} FP={r['fp']:2d} FN={r['fn']:2d} | "
                  f"Prec={r['precision']:.3f} Rec={r['recall']:.3f}{marker}")

        all_results[ev_name] = ev_results

    # Summary
    print(f"\n{'='*65}")
    print(" GAIN ANALYSIS")
    print(f"{'='*65}")

    for ev_name, ev_res in all_results.items():
        rec_name = EVENTS[ev_name]["recommended"]
        baseline_score = ev_res["conservative_unified"]["score"]
        rec_score = ev_res[rec_name]["score"]
        delta = rec_score - baseline_score
        emoji = "+" if delta > 0 else ("=" if delta == 0 else "")
        print(f"  {ev_name}: {rec_name} vs conservative_unified = {delta:+d} pts ({emoji}{'GAIN' if delta > 0 else 'NO CHANGE'})")

    total_baseline = sum(ev_res["conservative_unified"]["score"] for ev_res in all_results.values())
    total_dual = sum(
        all_results[ev]["competition_en"]["score"] if "EN" in ev else all_results[ev]["competition_fr"]["score"]
        for ev in all_results
    )
    print(f"\n  TOTAL COMBINED: conservative={total_baseline:+d} | dual={total_dual:+d} | delta={total_dual-total_baseline:+d}")

    # Export
    out_dir = Path("artifacts/competition")
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "dual_profile_benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Dual Profiles EN/FR\n\n")
        f.write(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("**Scoring** : `+2 TP` / `-2 FN` / `-6 FP`\n\n")

        for ev_name, ev_res in all_results.items():
            rec_name = EVENTS[ev_name]["recommended"]
            f.write(f"## {ev_name}\n\n")
            f.write("| Profil | Seuil | Score | TP | FP | FN | Precision | Recall |\n")
            f.write("|--------|-------|-------|----|----|----|-----------|--------|\n")
            for pname, r in ev_res.items():
                marker = " **◄**" if r["recommended"] else ""
                f.write(f"| {r['label']}{marker} | {r['threshold']:.2f} | **{r['score']:+d}** | "
                        f"{r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | {r['recall']:.3f} |\n")
            f.write("\n")

        f.write("## Gain Combiné\n\n")
        f.write(f"| Stratégie | Event 30 (EN) | Event 31 (FR) | **Total** |\n")
        f.write(f"|-----------|--------------|--------------|----------|\n")
        f.write(f"| Conservative unifié (t=0.60) | {all_results.get('Event30_EN',{}).get('conservative_unified',{}).get('score',0):+d} "
                f"| {all_results.get('Event31_FR',{}).get('conservative_unified',{}).get('score',0):+d} "
                f"| **{total_baseline:+d}** |\n")
        f.write(f"| **Dual EN/FR** | {all_results.get('Event30_EN',{}).get('competition_en',{}).get('score',0):+d} "
                f"| {all_results.get('Event31_FR',{}).get('competition_fr',{}).get('score',0):+d} "
                f"| **{total_dual:+d}** |\n")
        delta = total_dual - total_baseline
        verdict = "GAIN VALIDÉ" if delta > 0 else "PAS DE GAIN" if delta == 0 else "RÉGRESSION"
        f.write(f"\n> **Delta : {delta:+d} points — {verdict}**\n")

    json_path = out_dir / "dual_profile_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "total_conservative": total_baseline,
            "total_dual": total_dual,
            "delta": total_dual - total_baseline,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Report: {md_path}")
    print(f"  Data:   {json_path}")


if __name__ == "__main__":
    main()
