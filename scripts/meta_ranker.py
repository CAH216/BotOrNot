#!/usr/bin/env python
"""
scripts/meta_ranker.py — Aide à la décision pour le Jour J
==========================================================
Le Meta-Ranker analyse le rapport d'inspection du dataset et vos
objectifs de compétition pour formuler une recommandation experte :
 1. Le profil de soumission idéal (conservative, balanced, aggressive)
 2. La liste des modules de features à activer.

Usage type:
  python scripts/meta_ranker.py --train data/train.csv --metric f1 --fp-risk high
  python scripts/meta_ranker.py --report report.json --metric precision

Options :
  --metric   : auroc, f1, precision, recall (défaut : f1)
  --fp-risk  : high, low, medium (défaut : high)
"""
import argparse
import sys
import os
import json
from pathlib import Path

# Tentative d'importer le script d'inspection si on passe un CSV direct
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from inspect_dataset import inspect
except ImportError:
    inspect = None

SEP = "=" * 60

def _banner(m): print(f"\n{SEP}\n  {m}\n{SEP}")


def rank(report: dict, metric: str, fp_risk: str, scoring: str = "f1") -> dict:
    """Logique décisionnelle du Meta-Ranker."""
    sig = report.get("signals", {})
    
    # ── 1. Analyse de la Richesse des Données ──
    has_text = sig.get("text", {}).get("found", False)
    avg_len = sig.get("text", {}).get("avg_length", 0)
    text_richness = "high" if has_text and avg_len > 30 else "low" if has_text else "none"

    has_ts = sig.get("timestamps", {}).get("found", False)
    granularity = sig.get("timestamps", {}).get("granularity", "unknown")
    ts_richness = "high" if has_ts and granularity in ["second", "minute"] else "low" if has_ts else "none"

    has_struct = sig.get("profile_features", {}).get("found", False)
    struct_avail = "high" if has_struct else "none"

    has_edges = "edges" in report
    
    # ── 2. Recommandation du Profil de Soumission (Validation Historique) ──
    # Suite aux apprentissages des Events 30 (Anglais/Riche) et 31 (Français/Bruité)
    fp_risk = fp_risk.lower()
    is_rich_signal = (text_richness == "high" and struct_avail == "high")
    is_noisy_or_sparse = (text_richness in ["low", "none"] or struct_avail == "none")

    # Scoring officiel : +2 TP / -2 FN / -6 FP → FP coûte 3x FN
    # Avec ce barème, conservative est quasi-toujours optimal
    if scoring == "official":
        if is_rich_signal and fp_risk == "low":
            profile = "balanced"
            reason_prof = ("Scoring officiel (+2/-2/-6) MAIS signal exceptionnellement riche. "
                          "Balanced reste prudent tout en maximisant le score.")
        else:
            profile = "conservative"
            reason_prof = ("Scoring officiel (+2/-2/-6) : chaque FP coûte 3× un FN. "
                          "Le profil conservative maximise le score officiel en minimisant les FP.")
    elif is_noisy_or_sparse or fp_risk == "high":
        profile = "conservative"
        reason_prof = "Dataset s'apparentant au profil Français/Bruité (Event 31). Priorité absolue à la limitation des Faux Positifs via l'Anti-FP fort."
    elif is_rich_signal and fp_risk == "low":
        profile = "aggressive"
        reason_prof = "Dataset s'apparentant au profil Anglais/Riche (Event 30). Signal fort, faible ambiguïté. Maximisation de la détection (Aggressive)."
    else:  
        profile = "conservative"
        reason_prof = "Dataset mixte ou inconnu. Le profil conservative est le défaut absolu pour minimiser les FP."

    # ── 3. Recommandation des Modules ──
    modules = {}
    
    # Tabular (Toujours)
    modules["tabular"] = {"action": "ACTIVER", "reason": "Baseline incontournable."}
    
    # Temporal
    if ts_richness == "high":
        modules["temporal"] = {"action": "ACTIVER", "reason": f"Timestamps fins trouvés ({granularity})."}
    elif ts_richness == "low":
        modules["temporal"] = {"action": "ACTIVER", "reason": "Timestamps trouvés (mais granularité faible)."}
    else:
        modules["temporal"] = {"action": "DÉSACTIVER", "reason": "Aucun timestamp."}
        
    # Text
    if text_richness in ["high", "low"]:
        modules["text_basic"] = {"action": "ACTIVER", "reason": "Texte disponible (TF-IDF basique)."}
    else:
        modules["text_basic"] = {"action": "DÉSACTIVER", "reason": "Aucun texte."}
        
    if text_richness == "high":
        modules["text_model"] = {"action": "ACTIVER", "reason": f"Texte long détecté (moyenne={avg_len} chars)."}
    else:
        modules["text_model"] = {"action": "DÉSACTIVER", "reason": "Texte introuvable ou trop court."}
        
    # Structural
    if struct_avail == "high":
        modules["structural"] = {"action": "ACTIVER", "reason": "Méta-profils présents (source, followers, etc.)."}
    else:
        modules["structural"] = {"action": "DÉSACTIVER", "reason": "Données structurelles manquantes."}
        
    # Coordination
    if ts_richness == "high" and struct_avail == "high":
        modules["coordination"] = {"action": "ACTIVER", "reason": "Synergie Structurel + Temporel fin."}
    else:
        modules["coordination"] = {"action": "DÉSACTIVER", "reason": "Manque de timestamps fins ou info profils."}
        
    # Relational
    if has_edges:
        modules["relational"] = {"action": "ACTIVER", "reason": "Fichier d'arêtes fourni."}
    else:
        modules["relational"] = {"action": "DÉSACTIVER", "reason": "Aucun graphe fourni."}
        
    return {
        "profile": profile,
        "profile_reason": reason_prof,
        "modules": modules
    }


def main():
    parser = argparse.ArgumentParser(description="Aide à la décision : recommandations stratégiques pour BotOrNot.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", help="Chemin vers le fichier de données (lance inspect_dataset.py auto).")
    group.add_argument("--report", help="Chemin vers un rapport généré par inspect_dataset.py (--output).")
    
    parser.add_argument("--edges", default=None, help="Chemin vers edges (valable avec --train).")
    parser.add_argument("--metric", choices=["auroc", "f1", "precision", "recall"], default="f1",
                        help="Métriques officielle de la compétition.")
    parser.add_argument("--fp-risk", choices=["low", "medium", "high"], default="high",
                        help="Pénalité/Risque si on bannit un vrai humain par erreur.")
    parser.add_argument("--scoring", choices=["f1", "official"], default="official",
                        help="Mode de scoring : f1 (classique) ou official (+2 TP / -2 FN / -6 FP).")
    
    args = parser.parse_args()
    
    # Récupérer le rapport
    if args.report:
        if not os.path.exists(args.report):
            sys.exit(f"❌ Rapport introuvable : {args.report}")
        with open(args.report, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        if not inspect:
            sys.exit("❌ Impossible d'importer inspect_dataset.py. Utilisez --report ou réparez les imports.")
        print("🔍 Inspection rapide du dataset en cours...")
        report = inspect(args.train, args.edges)
        
    # Lancer le ranker
    result = rank(report, args.metric, args.fp_risk, args.scoring)
    
    _banner("🧠 META-RANKER — RECOMMANDATIONS")
    
    if args.scoring == "official":
        print("  ⚠️  SCORING OFFICIEL : +2 TP / -2 FN / -6 FP")
        print("  ⚠️  Chaque Faux Positif coûte 3× un Faux Négatif !\n")
    print(f"  Métrique cible : {args.metric.upper()}")
    print(f"  Risque Faux Positifs : {args.fp_risk.upper()}\n")
    
    print("  🏆 PROFIL DE SOUMISSION CONSEILLÉ :")
    print(f"  → {result['profile'].upper()}")
    print(f"    💡 {result['profile_reason']}\n")
    
    print("  🧩 ARCHITECTURE MODULES CONSEILLÉE :")
    for mod, desc in result["modules"].items():
        icon = "✅" if desc["action"] == "ACTIVER" else "❌"
        print(f"  {icon} {mod:<14} : {desc['action']:<10} | {desc['reason']}")
        
    print(f"\n{SEP}")
    print("  ACTIONS :")
    print("  1. Ajustez configs/features.yaml selon les recommandations ci-dessus.")
    print(f"  2. Lancez `python scripts/submission_factory.py --train {args.train or 'data/train.csv'} --format official`")
    print(f"  3. Soumettez le fichier `BotOrNot.detections.XX.txt`")
    print(SEP)
    

if __name__ == "__main__":
    main()
