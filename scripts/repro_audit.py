#!/usr/bin/env python
"""
scripts/repro_audit.py
======================
Génère l'audit final de reproductibilité :
- Version de Python, OS, encodage.
- Dépendances (via pip freeze).
- Commandes exactes d'exécution.
- Chemins de sortie attendus.
- Hashes (SHA-256) des fichiers de configuration critiques.

Génère :
- REPRO_AUDIT.md
- artifacts/repro/repro_snapshot.json
"""

import sys
import os
import json
import hashlib
import platform
import pkg_resources
from datetime import datetime
from pathlib import Path

def file_sha256(path: str) -> str:
    if not os.path.exists(path):
        return "Not Found"
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()

def get_packages():
    return {d.key: d.version for d in pkg_resources.working_set}

def generate_audit():
    t0 = datetime.now()
    
    # OS & Environment
    env_info = {
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "default_encoding": sys.getdefaultencoding(),
        "windows_utf8_env": os.environ.get("PYTHONUTF8", "Not Set")
    }
    
    # Packages
    packages = get_packages()
    
    # Commandes Exactes Day J
    commands = {
        "inspect_dataset": "python scripts/inspect_dataset.py data/train.csv",
        "meta_ranker": "python scripts/meta_ranker.py --train data/train.csv --metric f1 --fp-risk high",
        "run_baseline": "python scripts/run_baseline.py --train data/train.csv",
        "run_benchmark (rentabilité)": "python scripts/cost_benefit_benchmark.py --train data/train.csv",
        "submission_factory": "python scripts/submission_factory.py --train data/train.csv --test data/test.csv"
    }
    
    # Chemins de sortie
    outputs = {
        "submit_conservative": "artifacts/submissions/submission_conservative.csv",
        "submit_balanced": "artifacts/submissions/submission_balanced.csv",
        "submit_aggressive": "artifacts/submissions/submission_aggressive.csv",
        "report_meta_ranker": "Stdout (Terminal) ou via argument",
        "report_top_cases": "artifacts/top_cases/top_cases_report.md",
        "repro_snapshot": "artifacts/repro/repro_snapshot.json"
    }
    
    # Fichiers critiques (Les YAML purement indicatifs sont retirés)
    config_dir = Path("configs")
    target_configs = [
        "golden_baseline.yaml",
        "features.yaml",
        "inference.yaml"
    ]
    
    # Scripts Python contenant de la logique décisionnelle jour J
    core_dir = Path("scripts")
    target_scripts = [
        "submission_factory.py",
        "meta_ranker.py"
    ]
    
    hashes = {}
    for c in target_configs:
        p = config_dir / c
        hashes[f"configs/{c}"] = file_sha256(str(p))
        
    for s in target_scripts:
        p = core_dir / s
        hashes[f"scripts/{s}"] = file_sha256(str(p))
        
    audit_data = {
        "metadata": {
            "generated_at": t0.isoformat(),
            "project": "BotOrNot"
        },
        "environment": env_info,
        "packages": packages,
        "scripts_commands": commands,
        "expected_outputs": outputs,
        "config_hashes_sha256": hashes
    }
    
    # --- EXPORT JSON ---
    out_dir = Path("artifacts/repro")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "repro_snapshot.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
        
    # --- EXPORT MD ---
    md_path = Path("REPRO_AUDIT.md")
    
    lines = [
        "# Audit de Reproductibilité — BotOrNot",
        f"**Date de génération** : {t0.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Environnement d'Exécution",
        f"- **OS** : {env_info['os_system']} {env_info['os_release']} ({env_info['os_version']})",
        f"- **Python** : {env_info['python_version'].split()[0]}",
        f"- **Encodage par défaut** : `{env_info['default_encoding']}`",
        f"- **Variable Windows `PYTHONUTF8`** : `{env_info['windows_utf8_env']}` (Crucial pour la lecture des emojis sur Windows)",
        "",
        "## 2. Commandes d'Exécution (DAY J)",
        "Voici les commandes exactes à lancer dans le bon ordre :"
    ]
    
    for name, cmd in commands.items():
        lines.append(f"### {name}")
        if platform.system() == "Windows":
            lines.append("```powershell")
            lines.append(f"$env:PYTHONUTF8=1; {cmd}")
            lines.append("```")
        else:
            lines.append("```bash")
            lines.append(cmd)
            lines.append("```")
            
    lines.extend([
        "",
        "## 3. Chemins de Sortie Attendus",
        "| Livrable | Chemin Relatif |",
        "|---|---|"
    ])
    for desc, pth in outputs.items():
        lines.append(f"| {desc} | `{pth}` |")
        
    lines.extend([
        "",
        "## 4. Intégrité des Configurations (SHA-256)",
        "Ces hash garantissent que les règles d'extraction n'ont pas été altérées entre l'audit et l'exécution.",
        "| Fichier | Statut | Hash SHA-256 |",
        "|---|---|---|"
    ])
    
    for c, h in hashes.items():
        stat = "✅ Présent" if h != "Not Found" else "❌ Absent"
        lines.append(f"| `{c}` | {stat} | `{h[:16]}...` |")
        
    lines.append("\n## 5. Dépendances Python Majeures (Extrait)")
    key_deps = ["pandas", "numpy", "scikit-learn", "lightgbm", "catboost", "pyyaml"]
    for d in key_deps:
        v = packages.get(d, "Non installé")
        lines.append(f"- **{d}** : {v}")
        
    lines.append("\n*(Inventaire complet disponible dans `artifacts/repro/repro_snapshot.json`)*")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"✅ Audit généré :")
    print(f"   - {json_path}")
    print(f"   - {md_path}")

if __name__ == "__main__":
    generate_audit()
