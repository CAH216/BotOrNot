# 🤖 BotOrNot — Pipeline de Détection de Bots

Pipeline **modulaire** et **robuste** pour détecter les faux comptes sur les réseaux sociaux.  
Optimisé pour le **scoring officiel** : `+2 TP` / `-2 FN` / **`-6 FP`**

---

## ⚡ Démarrage rapide (Jour J)

### 1. Installer les dépendances

```bash
pip install pandas numpy scikit-learn lightgbm catboost joblib pyyaml pytest
```

### 2. Lancer le pipeline complet (format officiel)

```powershell
$env:PYTHONUTF8=1; python scripts/submission_factory.py --train data/train.csv --test data/test.csv --format official
```

Sortie : `BotOrNot.detections.{conservative,balanced,aggressive}.txt` — un user ID par ligne.

### 3. Évaluer localement avec le scoring officiel

```powershell
$env:PYTHONUTF8=1; python scripts/official_score.py --predicted predictions.txt --truth bots.txt
```

---

## 🏆 Scoring Officiel

| Événement | Gain/Pénalité |
|-----------|--------------|
| **Vrai Positif** (bot détecté) | **+2** |
| **Faux Négatif** (bot raté) | **-2** |
| **Faux Positif** (humain accusé) | **-6** |

> ⚠️ **1 Faux Positif coûte 3 bots ratés.** Le profil `conservative` est le choix par défaut.

## 📊 Benchmarks Validés

| Dataset | Conservative | Balanced | Aggressive |
|---------|-------------|----------|------------|
| **Event 30 (EN)** — 275 users | +68 | **+72** ⭐ | +72 |
| **Event 31 (FR)** — 129 users | **+6** ⭐ | 0 ❌ | +4 |

---

## 📁 Structure du projet

```
BotOrNot/
│
├── data/                        # Données (non versionnées)
├── dataset/                     # Datasets historiques (Events 30, 31)
│
├── configs/
│   ├── default.yaml             # Config globale
│   ├── features.yaml            # Modules de features
│   ├── competition_profile.yaml # Format officiel compétition
│   ├── models.yaml              # Hyperparamètres modèles
│   └── cv.yaml                  # Stratégie de validation croisée
│
├── src/
│   ├── data/                    # Chargement et profiling
│   │   └── adapters/            # Framework d'adaptateurs (historical, generic, twibot22)
│   ├── preprocessing/           # Normalisation colonnes, dates, texte
│   ├── features/                # Modules de features
│   ├── models/                  # Modèles ML (LightGBM, CatBoost, LR…)
│   ├── evaluation/              # Métriques et analyse d'erreurs
│   ├── inference/               # Prédiction, anti-FP, export
│   └── cli/                     # Point d'entrée pipeline complet
│
├── scripts/
│   ├── submission_factory.py    # 🏭 Générateur de soumissions (3 profils)
│   ├── official_score.py        # 🏆 Scoring officiel (+2/-2/-6)
│   ├── competition_benchmark.py # 📊 Benchmark format officiel
│   ├── meta_ranker.py           # 🧠 Recommandation de profil
│   ├── inspect_dataset.py       # 🔍 Inspection rapide d'un dataset
│   ├── run_baseline.py          # 🚀 Pipeline d'urgence
│   ├── run_cutdown.py           # ⚡ Version ultra-légère
│   ├── historical_benchmark.py  # 📈 Benchmark Events 30/31
│   ├── adversarial_robustness.py# 🛡️ Audit adversarial (4 scénarios)
│   ├── repro_audit.py           # 🔒 Audit de reproductibilité
│   └── ablation.py              # 🔬 Ablation study
│
├── tests/                       # Tests unitaires + non-régression
│
├── PLAYBOOK.md                  # Guide opérationnel Jour J
├── DECISION_CARD.md             # Aide-mémoire rapide
├── RULES.md                     # Règles de développement
└── COMPETITION_FREEZE.md        # Gel du code pour compétition
```

---

## 🏭 Submission Factory

Le script principal de la compétition. Produit 3 soumissions en un seul run.

```powershell
# Mode compétition (sortie .txt)
$env:PYTHONUTF8=1; python scripts/submission_factory.py \
    --train data/train.csv --test data/test.csv \
    --format official --team-name BotOrNot

# Mode classique (sortie .csv)
$env:PYTHONUTF8=1; python scripts/submission_factory.py \
    --train data/train.csv --test data/test.csv
```

**Profils disponibles :**

| Profil | Seuil | Anti-FP | Usage |
|--------|-------|---------|-------|
| `conservative` | 0.60 | Fort | **Défaut** — Minimise les FP |
| `balanced` | F1-auto | Modéré | Compromis F1 / FP |
| `aggressive` | 0.38 | Désactivé | Recall maximal (⚠️ dangereux avec -6 FP) |

---

## 🧠 Meta-Ranker

Analyse le dataset et recommande le profil optimal.

```powershell
$env:PYTHONUTF8=1; python scripts/meta_ranker.py --train data/train.csv --scoring official
```

---

## 🛡️ Audit de Robustesse Adversariale

Teste le pipeline contre 4 types de bots furtifs :

| Scénario | Description | Résultat |
|----------|-------------|---------|
| **Sleeper Bot** | Bursts + silence 8-12h | 🟢 SAFE |
| **Jitter Bot** | Délais aléatoires | 🟢 SAFE |
| **LLM Bot** | Texte fluide, faible volume | 🟢 SAFE |
| **Camouflage Bot** | Imitation complète profil humain | 🟢 SAFE |

```powershell
$env:PYTHONUTF8=1; python scripts/adversarial_robustness.py
```

---

## 🧩 Modules de features

| Module | Activé par défaut | Ce qu'il produit |
|--------|-------------------|------------------|
| **Tabular** | ✅ | tweet_count, z_score, bio/username stats |
| **Temporal** | ✅ | IPT moyen/std/CV, entropie horaire, ratio nuit |
| **Text Basic** | ✅ | Longueur, URLs, mentions, hashtags, diversité |
| **Structural** | ❌ (pas dans format officiel) | Anomalies de format, patterns d'IDs |
| **Relational** | ❌ (pas d'edges) | Degree, clustering, reciprocity |
| **Embeddings** | ❌ (pas de GPU garanti) | Sentence-transformers |

---

## 🔍 Inspection de dataset

```bash
python scripts/inspect_dataset.py data/train.csv
```

Détecte automatiquement : colonnes, timestamps, texte, labels, types de features.

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

---

## 📦 Dépendances

```
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
lightgbm >= 3.3          # recommandé
catboost >= 1.2           # optionnel
joblib >= 1.2
pyyaml >= 6.0
pytest >= 7.0
```

---

*Projet BotOrNot — Pipeline anti-fraude modulaire. Scoring officiel : +2 TP / -2 FN / -6 FP.*
