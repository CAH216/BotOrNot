# BotOrNot : Competition Freeze (Master Pack)

Ce document consigne l'état de congélation de la base de code (Code Freeze) et constitue le point de vérité unique (Single Source of Truth) pour l'exécution lors de la compétition BotOrNot.

Toute R&D exploratoire est suspendue. L'architecture a été validée via les événements pré-compétition (Events 30 et 31).

---

## ⚠️ Scoring Officiel

```
+2  Vrai Positif   (bot correctement détecté)
-2  Faux Négatif   (bot raté)
-6  Faux Positif   (humain accusé à tort)
```

> **Avec le score officiel (+2 / -2 / -6), un faux positif coûte 3× un vrai positif. La stratégie par défaut doit donc minimiser les FP avant toute autre considération.**

## 1. Politique Finale des Profils

**`conservative` est le profil par DÉFAUT ABSOLU.**

* **Conservative** (DÉFAUT) : Priorité absolue Anti-FP. Imposé par le scoring asymétrique (-6 FP). Obligatoire sur les datasets bruités, français, inconnus, ou à risque FP non négligeable.
* **Balanced** : Compromis F1 / FP. Envisageable **uniquement** sur dataset anglais avec signal fort et faible ambiguïté.
* **Aggressive** : Recall maximal. **Déconseillé** avec le scoring officiel — chaque FP supplémentaire coûte 3 bots ratés.

### Résumé de Politique (5 lignes)

1. **Français / bruité / inconnu → conservative**
2. **Doute sur la nature des données → conservative**
3. **Anglais + signal fort + faible ambiguïté → balanced (ou aggressive)**
4. **Scoring officiel + risque FP non négligeable → conservative**
5. **En cas de doute → conservative**

## 2. Format Officiel

* **Entrée** : `dataset.posts&users.XX.json`
* **Sortie** : `BotOrNot.detections.XX.txt` — un user ID par ligne
* **Pas de** : followers_count, following_count, source, edges

## 3. Commandes Exactes Jour J

🚨 **Préalable Windows Absolu** : Toujours préfixer avec `$env:PYTHONUTF8=1;`

### A. Baseline Officielle (Submission Factory — Format Compétition)
```powershell
$env:PYTHONUTF8=1; python scripts/submission_factory.py --train data/train.csv --test data/test.csv --format official
```

### B. Cutdown (Plan B / Urgence)
```powershell
$env:PYTHONUTF8=1; python scripts/run_cutdown.py --train data/train.csv --test data/test.csv --profile conservative
```

### C. Vérification Score Officiel
```powershell
$env:PYTHONUTF8=1; python scripts/official_score.py --predicted predictions.txt --truth bots.txt
```

## 4. Topologie des Sorties Attendues

Après exécution avec `--format official` :

- `BotOrNot.detections.conservative.txt` ← **SOUMISSION PAR DÉFAUT**
- `BotOrNot.detections.balanced.txt`
- `BotOrNot.detections.aggressive.txt`
- `submission_conservative.csv` (interne)
- `submission_balanced.csv` (interne)
- `submission_aggressive.csv` (interne)
- `factory_report.json`

## 5. Manifestes Approuvés & Audités
- Aide Mémoire Rapide : `DECISION_CARD.md`
- Workflow Stratégique : `PLAYBOOK.md`
- Audit de Reproductibilité : `REPRO_AUDIT.md`
- Config Format Officiel : `configs/competition_profile.yaml`

Aucun autre changement majeur ne sera opéré. Bon courage.
