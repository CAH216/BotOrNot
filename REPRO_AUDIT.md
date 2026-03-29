# Audit de Reproductibilité — BotOrNot
**Date de génération** : 2026-03-28 13:35:45

## 1. Environnement d'Exécution
- **OS** : Windows 10 (10.0.26200)
- **Python** : 3.11.4
- **Encodage par défaut** : `utf-8`
- **Variable Windows `PYTHONUTF8`** : `1` (Crucial pour la lecture des emojis sur Windows)

## 2. Commandes d'Exécution (DAY J)
Voici les commandes exactes à lancer dans le bon ordre :
### inspect_dataset
```powershell
$env:PYTHONUTF8=1; python scripts/inspect_dataset.py data/train.csv
```
### meta_ranker
```powershell
$env:PYTHONUTF8=1; python scripts/meta_ranker.py --train data/train.csv --metric f1 --fp-risk high
```
### run_baseline
```powershell
$env:PYTHONUTF8=1; python scripts/run_baseline.py --train data/train.csv
```
### run_benchmark (rentabilité)
```powershell
$env:PYTHONUTF8=1; python scripts/cost_benefit_benchmark.py --train data/train.csv
```
### submission_factory
```powershell
$env:PYTHONUTF8=1; python scripts/submission_factory.py --train data/train.csv --test data/test.csv
```

## 3. Chemins de Sortie Attendus
| Livrable | Chemin Relatif |
|---|---|
| submit_conservative | `artifacts/submissions/submission_conservative.csv` |
| submit_balanced | `artifacts/submissions/submission_balanced.csv` |
| submit_aggressive | `artifacts/submissions/submission_aggressive.csv` |
| report_meta_ranker | `Stdout (Terminal) ou via argument` |
| report_top_cases | `artifacts/top_cases/top_cases_report.md` |
| repro_snapshot | `artifacts/repro/repro_snapshot.json` |

## 4. Intégrité des Configurations (SHA-256)
Ces hash garantissent que les règles d'extraction n'ont pas été altérées entre l'audit et l'exécution.
| Fichier | Statut | Hash SHA-256 |
|---|---|---|
| `configs/golden_baseline.yaml` | ✅ Présent | `49522352622ff818...` |
| `configs/features.yaml` | ✅ Présent | `d08083d5de01f43f...` |
| `configs/inference.yaml` | ✅ Présent | `78a5d8e329724483...` |
| `scripts/submission_factory.py` | ✅ Présent | `335330b6b4c4ae13...` |
| `scripts/meta_ranker.py` | ✅ Présent | `630b946cae0c842c...` |

## 5. Dépendances Python Majeures (Extrait)
- **pandas** : 3.0.1
- **numpy** : 1.26.4
- **scikit-learn** : 1.8.0
- **lightgbm** : 4.6.0
- **catboost** : Non installé
- **pyyaml** : 6.0.3

*(Inventaire complet disponible dans `artifacts/repro/repro_snapshot.json`)*