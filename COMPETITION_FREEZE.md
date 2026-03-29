# BotOrNot : Competition Freeze (Master Pack)

Ce document consigne l'état de congélation de la base de code (Code Freeze) et constitue le point de vérité unique (Single Source of Truth) pour l'exécution lors de la compétition BotOrNot.

Toute R&D exploratoire est suspendue. L'architecture a été validée via les événements pré-compétition (Events 30 et 31).

---

## 1. Profils Disponibles & Logique Décisionnelle
La sélection du profil dépend de la nature des données testées, validée sur nos benchmarks linguistiques.
Le choix du profil s'effectuera via la recommandation automatique du `meta_ranker.py`.

* **Balanced** (DÉFAUT) : Compromis optimal. Filtre Anti-FP modéré. Utilisé en l'absence de signaux extrêmes.
* **Conservative** : Priorité absolue Anti-FP. Obligatoire sur les datasets bruités, rares en texte, type **FRANÇAIS**. (Basé sur Event 31).
* **Aggressive** : Priorité recall maximal. Sûr à utiliser sur les gros datasets purs, type **ANGLAIS**. (Basé sur Event 30).

## 2. Baselines Officielles & Commandes Exactes Jour J

🚨 **Préalable Windows Absolu** : Toujours préfixer la console avec `$env:PYTHONUTF8=1;` pour éviter les plantages lors de la lecture des emojis.

### A. La Baseline Officielle (Submission Factory)
L'Usine à soumissions exécute le pipeline ML central, croise les features Textuelles/Structurelles/Temporelles et filtre au travers de l'AntiFP pour produire les 3 CSV préconfigurés. 

```powershell
$env:PYTHONUTF8=1; python scripts/submission_factory.py --train data/train.csv --test data/test.csv
```

### B. Cutdown Baseline (Plan B / Sécurité)
Si le serveur manque de RAM ou que l'usine plante face à un volume imprévu, la fonction "Cutdown" abandonne l'AntiFP pour produire une prédiction brute, rapide et légère du profil par défaut.

```powershell
$env:PYTHONUTF8=1; python scripts/run_cutdown.py --train data/train.csv --test data/test.csv --profile balanced
```

## 3. Topologie des Sorties Attendues

Après l'exécution de l'outil cible `submission_factory.py`, voici STRICTEMENT les fichiers attendus dans le dossier `artifacts/submissions/` : 

- `submission_conservative.csv` (Probabilités recalibrées)
- `submission_conservative_meta.json` (Traceabilité)
- `submission_balanced.csv` (Probabilités recalibrées)
- `submission_balanced_meta.json` (Traceabilité)
- `submission_aggressive.csv` (Probabilités brutes bas-seuil)
- `submission_aggressive_meta.json` (Traceabilité)
- `factory_report.json` (Synthèse croisée des métriques internes)

*Vous chargerez le fichier CSV correspondant à la recommandation émise.*

## 4. Manifestes Approuvés & Audités
- Aide Mémoire Rapide de l'opérateur : `DECISION_CARD.md`.
- Workflow Analytique Stratégique : `PLAYBOOK.md`.
- Matrice d'intégrité (Check SHA-256) : `REPRO_AUDIT.md`.

Aucun autre changement majeur de fichier ne sera opéré. Bon courage.
