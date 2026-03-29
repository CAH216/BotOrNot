#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/official_score.py
==========================
Évaluation locale avec le scoring officiel de la compétition BotOrNot.

Scoring :
  +2  par Vrai Positif  (bot correctement détecté)
  -2  par Faux Négatif  (bot raté)
  -6  par Faux Positif  (humain accusé à tort)

Usage :
  python scripts/official_score.py --predicted preds.txt --truth bots.txt --all-users users.txt
  python scripts/official_score.py --predicted preds.txt --truth bots.txt --total-users 275

Entrées :
  --predicted   : fichier texte avec un user ID par ligne (bots prédits)
  --truth       : fichier texte avec un user ID par ligne (vrais bots)
  --all-users   : (optionnel) fichier texte avec tous les user IDs
  --total-users : (optionnel) nombre total d'utilisateurs (si --all-users absent)
"""

import argparse
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Scoring Officiel
# ─────────────────────────────────────────────────────────────

TP_WEIGHT = +2
FN_WEIGHT = -2
FP_WEIGHT = -6


def load_ids(filepath):
    """Charge un fichier texte avec un ID par ligne."""
    p = Path(filepath)
    if not p.exists():
        sys.exit(f"❌ Fichier introuvable : {filepath}")
    with open(p, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def compute_official_score(predicted_bots, true_bots, all_users=None):
    """
    Calcule le score officiel de la compétition.

    Args:
        predicted_bots: set of user IDs prédits comme bots
        true_bots:      set of user IDs réellement bots
        all_users:      set of all user IDs (optionnel, pour calculer TN)

    Returns:
        dict avec TP, FP, FN, TN, score, score_max, efficiency
    """
    tp = len(predicted_bots & true_bots)
    fp = len(predicted_bots - true_bots)
    fn = len(true_bots - predicted_bots)

    score = (TP_WEIGHT * tp) + (FN_WEIGHT * fn) + (FP_WEIGHT * fp)
    score_max = TP_WEIGHT * len(true_bots)  # score parfait = tous les bots détectés, 0 FP

    tn = None
    if all_users is not None:
        true_humans = all_users - true_bots
        tn = len(true_humans - predicted_bots)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_predicted": len(predicted_bots),
        "n_true_bots": len(true_bots),
        "n_total_users": len(all_users) if all_users else None,
        "score": score,
        "score_max": score_max,
        "efficiency": round(score / score_max, 4) if score_max > 0 else 0,
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "fp_cost": FP_WEIGHT * fp,
        "fn_cost": FN_WEIGHT * fn,
        "tp_gain": TP_WEIGHT * tp,
    }


def print_report(result):
    """Affiche le rapport de scoring officiel."""
    sep = "=" * 55
    print(f"\n{sep}")
    print("  🏆 SCORING OFFICIEL — BotOrNot Competition")
    print(sep)

    print(f"\n  📊 Détails :")
    print(f"     Bots prédits  : {result['n_predicted']}")
    print(f"     Vrais bots    : {result['n_true_bots']}")
    if result['n_total_users']:
        print(f"     Total users   : {result['n_total_users']}")

    print(f"\n  ✅ Vrais Positifs  (TP) : {result['tp']:>5}  →  +{result['tp_gain']}")
    print(f"  ❌ Faux Positifs  (FP) : {result['fp']:>5}  →  {result['fp_cost']}")
    print(f"  ⚠️  Faux Négatifs (FN) : {result['fn']:>5}  →  {result['fn_cost']}")
    if result['tn'] is not None:
        print(f"  ✓  Vrais Négatifs (TN) : {result['tn']:>5}")

    print(f"\n  {'─'*40}")
    print(f"  🎯 SCORE OFFICIEL : {result['score']:>+6}")
    print(f"     Score Maximum  : {result['score_max']:>+6}")
    print(f"     Efficacité     : {result['efficiency']:.1%}")
    print(f"\n  Precision : {result['precision']:.4f}")
    print(f"  Recall    : {result['recall']:.4f}")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="Scoring officiel BotOrNot (+2 TP / -2 FN / -6 FP)")
    parser.add_argument("--predicted", required=True, help="Fichier .txt des bots prédits (1 ID/ligne)")
    parser.add_argument("--truth", required=True, help="Fichier .txt des vrais bots (1 ID/ligne)")
    parser.add_argument("--all-users", default=None, help="Fichier .txt de tous les user IDs (optionnel)")
    parser.add_argument("--total-users", type=int, default=None, help="Nombre total d'utilisateurs (optionnel)")
    args = parser.parse_args()

    predicted = load_ids(args.predicted)
    truth = load_ids(args.truth)

    all_users = None
    if args.all_users:
        all_users = load_ids(args.all_users)
    elif args.total_users:
        # If we only know the total, we can still compute TN approximately
        all_users = truth | predicted  # minimum set

    result = compute_official_score(predicted, truth, all_users)
    print_report(result)


if __name__ == "__main__":
    main()
