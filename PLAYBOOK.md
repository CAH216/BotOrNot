# BotOrNot : Playbook Jour J

Ce guide opérationnel documente l'esprit du pipeline et la séquence des actions pour le Jour J.

## ⚠️ Scoring Officiel

```
+2 par Vrai Positif  (bot correctement détecté)
-2 par Faux Négatif  (bot raté)
-6 par Faux Positif  (humain accusé à tort)
```

**Un seul FP coûte autant que 3 bots ratés.** Le profil `conservative` est le choix par défaut absolu.

## Format Officiel

- **Entrée** : `dataset.posts&users.XX.json` (contient `users[]` et `posts[]`)
- **Sortie** : fichier texte `BotOrNot.detections.XX.txt` — un user ID par ligne
- **Champs disponibles** :
  - Posts : `text`, `created_at`, `id`, `author_id`, `lang`
  - Users : `id`, `username`, `name`, `description`, `location`, `tweet_count`, `z_score`
- **Pas de** `followers_count`, `following_count`, `source`, `edges`

## Philosophie de la Détection

Notre approche repose sur l'assemblage de signaux multi-famille (tabular, temporal, textuel) et l'application d'un **Anti-FP Filter** post-modèle.

* **Scoring officiel (+2/-2/-6)** ➜ **Conservative** est quasi-toujours optimal.
* **Signal exceptionnellement riche + 0 ambiguïté** ➜ Envisager `balanced` (jamais `aggressive`).
* **En cas de doute** ➜ `conservative`.

## Séquence Opérationnelle Exacte

Veuillez lancer systématiquement la variable environnementale de préfixe pour le support des émojis UTF-8 Windows : `$env:PYTHONUTF8=1;`

**1. Analyser les Signaux & Recommander (Meta Ranker)**
```powershell
$env:PYTHONUTF8=1; python scripts/meta_ranker.py --train data/train.csv --scoring official
```

**2. Lancement Complet (Submission Factory — Format Officiel)**
Génère les fichiers `.txt` (format compétition) + `.csv` (interne).
```powershell
$env:PYTHONUTF8=1; python scripts/submission_factory.py --train data/train.csv --test data/test.csv --format official --team-name BotOrNot
```

**3. Vérification Locale (Score Officiel)**
Compare une prédiction au ground truth avec le barème officiel :
```powershell
$env:PYTHONUTF8=1; python scripts/official_score.py --predicted predictions.txt --truth bots.txt
```

**4. Plan de Secours Absolu (Cutdown)**
Si la mémoire sature :
```powershell
$env:PYTHONUTF8=1; python scripts/run_cutdown.py --train data/train.csv --test data/test.csv --profile conservative
```
