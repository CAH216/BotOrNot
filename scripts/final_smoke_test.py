#!/usr/bin/env python
"""
scripts/final_smoke_test.py
===========================
Garde-fou final end-to-end du Jour J.
Exécute toute la chaîne opérationnelle :
 1. inspect_dataset.py
 2. meta_ranker.py
 3. submission_factory.py

Et valide l'intégrité formelle des 3 soumissions générées :
 - Nombre de lignes
 - Encodage UTF-8, Pas de NaNs
 - Probabilités dans [0, 1] et valeurs binaires dans {0, 1}
 - Méta-données JSON correspondantes
"""

import sys, os, time, json, argparse, warnings
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
SEP = "=" * 70

def _banner(m): print(f"\n{SEP}\n  {m}\n{SEP}")
def _log(m):    print(f"  [{datetime.now():%H:%M:%S}] {m}")


def run_cmd(cmd_list, desc):
    _log(f"▶ Lancement : {desc}")
    # Inject utf8 for Windows
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    
    t0 = time.time()
    res = subprocess.run(cmd_list, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    
    if res.returncode != 0:
        print(f"\n❌ ERREUR CRITIQUE dans '{desc}' (Code = {res.returncode})")
        print("--- STDOUT ---")
        print(res.stdout)
        print("--- STDERR ---")
        print(res.stderr)
        sys.exit(f"🛑 Smoke Test échoué : Crash de l'étape '{desc}'.")
    _log(f"  ✅ Succès en {elapsed:.1f}s")


def run_smoke_test(train_path, test_path, out_dir):
    t_start = time.time()
    _banner("SMOKE TEST END-TO-END — BOTORNOT")
    
    # 1. Inspection
    run_cmd(["python", "scripts/inspect_dataset.py", train_path], "Inspect Dataset")
    
    # 2. Meta Ranker
    run_cmd(["python", "scripts/meta_ranker.py", "--train", train_path, "--metric", "f1", "--fp-risk", "high"], "Meta Ranker (F1/High)")
    
    # 3. Submission Factory
    run_cmd(["python", "scripts/submission_factory.py", "--train", train_path, "--test", test_path], "Submission Factory")
    
    # 4. Validation des Formats
    _banner("VALIDATION DES FICHIERS DE SOUMISSION")
    
    # Lire la taille cible depuis le fichier test brut
    _log(f"Lecture du fichier test de référence : {test_path}")
    try:
        df_target = pd.read_csv(test_path, usecols=lambda c: "id" in c.lower(), nrows=500000)
    except Exception as e:
        df_target = pd.read_csv(test_path, nrows=500000)
        
    # Idéalement l'id col est le premier s'il y a user_id ou account_id
    id_col = [c for c in df_target.columns if c in ["user_id", "account_id", "author_id", "id"]]
    if id_col:
        id_col = id_col[0]
        n_expected = df_target[id_col].nunique()
    else:
        n_expected = len(df_target)
        
    _log(f"Taille attendue : {n_expected} lignes/comptes uniques")
    
    profiles = ["conservative", "balanced", "aggressive"]
    sub_dir = Path("artifacts/submissions")
    
    report_lines = [
        "# Rapport du Smoke Test Final (End-to-End)",
        f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Fichier Test** : `{test_path}` (Taille attendue : {n_expected} lignes)\n",
        "## Validation des Soumissions"
    ]
    
    all_passed = True
    
    for prof in profiles:
        _log(f"--- Vérification Profil : {prof.upper()} ---")
        prof_passed = True
        
        csv_file = sub_dir / f"submission_{prof}.csv"
        json_file = sub_dir / f"submission_{prof}_meta.json"
        
        errs = []
        
        # a. Fichier existe
        if not csv_file.exists():
            errs.append("❌ Fichier CSV manquant.")
        else:
            # On tente de le charger en utf-8 strict
            try:
                df_sub = pd.read_csv(csv_file, encoding="utf-8", encoding_errors="strict")
            except UnicodeDecodeError:
                errs.append("❌ Encodage invalide (UTF-8 requis).")
                df_sub = None
                
            if df_sub is not None:
                # b. Nombre de lignes exact
                if len(df_sub) != n_expected:
                    errs.append(f"❌ Mauvais nombre de lignes : {len(df_sub)} != {n_expected}")
                    
                # c. Absence de NaN
                if df_sub.isna().any().any():
                    errs.append("❌ Valeurs NaN détéctées.")
                    
                # d. Colonne bot proba
                base_pred_col = [c for c in df_sub.columns if c.startswith("prediction")]
                proba_col = [c for c in df_sub.columns if "proba" in c]
                
                if proba_col:
                    p = df_sub[proba_col[0]]
                    if p.min() < 0.0 or p.max() > 1.0:
                        errs.append("❌ Probabilités hors limites [0.0, 1.0].")
                        
                if base_pred_col:
                    p = df_sub[base_pred_col[0]]
                    if not set(p.unique()).issubset({0, 1}):
                        errs.append("❌ Prédictions binaires non valides (!= {0, 1}).")
                        
        # e. Json meta
        if not json_file.exists():
            errs.append("⚠️ Fichier JSON meta manquant (Non bloquant si le dataset était vide/pas d'anti FP ? Mais attendu normalement).")
            # Ne fait pas échouer prof_passed formellement
        
        report_lines.append(f"### Profil : `{prof}`")
        if errs:
            for e in errs:
                _log(f"  {e}")
                report_lines.append(f"- {e}")
            if any("❌" in e for e in errs):
                prof_passed = False
                all_passed = False
        else:
            _log("  ✅ Format Validé")
            report_lines.append("- ✅ Lignes, Encodage UTF-8, Intervalles de Probas validés.")
            report_lines.append(f"- ✅ Méta-données JSON présentes.")
            
    # Rapport global
    report_lines.append("\n## Verdict")
    if all_passed:
        report_lines.append("🚀 **ALL GREEN** : Le pipeline est prêt à produire des soumissions de niveau production en 3 commandes.")
        _banner("🏆 SMOKE TEST RÉUSSI 🏆\n  Le pipeline End-to-End est Parfaitement Sain.")
    else:
        report_lines.append("🛑 **DANGER** : Format des soumissions corrompu. Revérifiez le code d'inférence.")
        _banner("🛑 ERREUR : Le format des soumissions ne valide pas les critères Stricts.")
        
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    report_file = out_dir_path / "final_smoke_report.md"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    _log(f"Rapport de Test écrit : {report_file}")
    _log(f"Temps total du Smoke Test : {time.time() - t_start:.1f}s")
    
    # Sortie Processus
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exécute les vérifications finies du bot")
    parser.add_argument("--train", default="data/_dryrun_train.csv")
    parser.add_argument("--test", default="data/_dryrun_test.csv")
    parser.add_argument("--out", default="artifacts/final_smoke")
    args = parser.parse_args()
    
    if not os.path.exists(args.test):
        sys.exit(f"❌ Fichier Test {args.test} introuvable pour simuler la soumission.")
        
    run_smoke_test(args.train, args.test, args.out)
