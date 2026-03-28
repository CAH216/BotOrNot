# 🗓️ PLAYBOOK DAY J — BotOrNot

> **Objectif :** Produire une soumission compétitive en moins de 30 minutes à partir de données inconnues.

---

## ⏱️ Étape 1 — Les 10 premières minutes

### Minute 0–2 : Inspecter le dataset
```bash
python scripts/inspect_dataset.py data/train.csv
python scripts/inspect_dataset.py data/train.csv --edges data/edges.csv  # si graphe
```

**Ce que tu dois noter :**
- [ ] Colonnes présentes (id, label, timestamps, texte, followers…)
- [ ] Taux de valeurs manquantes
- [ ] Unité de prédiction : compte ou post ?
- [ ] Modules recommandés par le script

### Minute 2 : La Decision Gate Automatique (Meta-Ranker)
Plutôt que de choisir manuellement votre stratégie, laissez le **Meta-Ranker** croiser la richesse de vos données avec les règles de la compétition.
```bash
python scripts/meta_ranker.py --train data/train.csv --metric f1 --fp-risk high
```
→ **Output direct :** Quel(s) module(s) activer dans `configs/features.yaml` et quel **Profil** soumettre à la fin (`conservative`, `balanced` ou `aggressive`).

---

### Minute 2–5 : Lancer le benchmark
```bash
python scripts/benchmark.py --train data/train.csv --cv-folds 3 --out artifacts/benchmark
```

→ Il te dit quelle combinaison de features donne le meilleur AUROC.

---

### Minute 5–10 : Lancer le baseline
```bash
python scripts/run_baseline.py --train data/train.csv --test data/test.csv --cv-folds 3
```

→ Soumission immédiate dans `artifacts/submissions/baseline.csv`.

> **Tu as déjà quelque chose à soumettre. La suite améliore ce score.**

---

## 🧩 Étape 2 — Choisir les bons modules

| Signal détecté | Module à activer | Impact attendu |
|---|---|---|
| Timestamps précis (< minute) | `temporal` Plan A | Très fort — IPT, burst, nuit |
| Timestamps date seulement | `temporal` Plan B | Moyen — posts/jour, densité |
| Texte court (tweets, bio) | `text_basic` | Moyen — URLs, mentions, #hashtags |
| Texte long + varié | `text_model` | Bon si diversité linguistique |
| Followers/following | `tabular` | Toujours actif — ratios FF, extrêmes |
| Source/API/client | `structural` | Ajouter si colonnes présentes |
| Graphe (edges.csv) | `relational` | Fort si disponible |
| Beaucoup de texte + GPU | `text_embeddings` | Secondaire, ignorer si temps court |

**Règle :** Active un module **seulement si la donnée correspondante existe en quantité suffisante** (> 20% non-null).

---

## 🏆 Étape 3 — Choisir le modèle

| Situation | Recommandation |
|---|---|
| Dataset > 1000 comptes | `lgbm` ou `catboost` |
| Dataset < 500 comptes | `lr` (moins de risque d'overfit) |
| Pas le temps | `lgbm` par défaut — toujours solide |
| Labels très déséquilibrés (< 10% bots) | Augmenter le seuil : `--threshold 0.4` |
| Labels très déséquilibrés (> 80% bots) | Descendre le seuil : `--threshold 0.3` |

---

## 📤 Étape 4 — Produire la soumission finale

### Option A — Factory auto (3 soumissions en 1 run, **recommandée**)
```bash
python scripts/submission_factory.py \
    --train data/train.csv \
    --test  data/test.csv \
    --model lgbm --cv-folds 5
```
→ Génère `submission_conservative.csv`, `submission_balanced.csv`, `submission_aggressive.csv` + `factory_report.json`.

### Option B — Ablation study rapide (quel module contribue le plus ?)
```bash
python scripts/ablation.py --train data/train.csv --cv-folds 3 --out artifacts/ablation
```
→ Compare 13 configurations et exporte le delta AUROC par rapport à l'ensemble.

### Option C — Baseline rapide (< 5 min)
```bash
python scripts/run_baseline.py \
    --train data/train.csv \
    --test data/test.csv \
    --model lgbm \
    --cv-folds 5 \
    --out artifacts/submissions/v1_final
```

### Option D — Après benchmark (meilleure combinaison)
```bash
python scripts/benchmark.py --train data/train.csv --out artifacts/benchmark
# Puis choisir le meilleur pipeline
python scripts/run_baseline.py --train data/train.csv --test data/test.csv \
    --model catboost --threshold 0.55
```

---

## ⚠️ Pièges à éviter

| Piège | Quoi faire |
|---|---|
| Fuites de données (leakage) | Le script utilise `GroupKFold` par compte — tu es protégé |
| Labels déséquilibrés | Tous les modèles ont `class_weight='balanced'` |
| Colonnes absentes dans le test | Les colonnes manquantes sont remplies avec `NaN` automatiquement |
| Bons power-users classés bots | Augmenter le seuil (`--threshold 0.55`) |
| Manque de temps | Utilise `--no-cv` pour aller encore plus vite |

---

## 📁 Fichiers générés

Après `run_baseline.py`, tu as :

```
artifacts/submissions/
├── v1_final.csv                     ← Soumission à déposer
├── v1_final_meta.json               ← AUC, F1, seuil, config
└── v1_final_feature_importances.csv ← Top features
```

**Vérification rapide avant de soumettre :**
```bash
python -c "import pandas as pd; df=pd.read_csv('artifacts/submissions/v1_final.csv'); print(df['label'].value_counts()); print(df.head())"
```

---

## Decision Gate — Avant de soumettre

> 💡 **RAPPEL :** Utilisez `python scripts/meta_ranker.py --train data/train.csv --metric <votre_metrique> --fp-risk <high/low>` pour générer automatiquement la recommandation en base des données réelles.

### 1. Vérifier la métrique officielle (Rappel manuel)

| Métrique annoncée | Profil recommandé | Logique |
|---|---|---|
| **AUROC** | `balanced` ou `aggressive` | AUROC n'est pas affectée par le seuil |
| **F1-score** | `balanced` | Seuil F1-optimal auto-calculé |
| **Précision** | `conservative` | Minimiser les FP |
| **Recall** | `aggressive` | Maximiser les vrais positifs |
| **Non précisée** | `balanced` | Meilleur compromis par défaut |

### 2. Vérifier le format de soumission

```bash
# Vérifier la structure du fichier généré
python -c "
import pandas as pd
df = pd.read_csv('artifacts/submissions/submission_balanced.csv')
print('Colonnes :', list(df.columns))
print('Lignes :', len(df))
print('Labels :', df.iloc[:, -1].value_counts().to_dict())
print('NaN :', df.isna().sum().sum())
print('Apercu :'); print(df.head(3))
"
```

**Checklist avant soumission :**
- [ ] Toutes les lignes du test sont présentes (même count que le test original)
- [ ] Pas de NaN dans la colonne label
- [ ] Les labels sont en 0/1 (ou proba si la compétition l'exige)
- [ ] L'ID compte correspond bien aux IDs du fichier de test
- [ ] Le fichier .csv est bien encodé (UTF-8 ou ASCII)

### 3. Choisir conservative / balanced / aggressive

```
Métrique = Précision → conservative (seuil 0.60)
Métrique = F1        → balanced    (seuil auto 0.45–0.65)
Métrique = Recall    → aggressive  (seuil 0.38)
Métrique = AUROC     → soumettre les 3, garder le balanced
Doute    → commencer par balanced, puis conservative si FP pénalisés
```

### 4. Fix Windows (si emoji bloque le terminal)

```powershell
# A lancer UNE FOIS dans PowerShell avant les commandes
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## Mode urgence absolue (< 5 minutes)

```bash
$env:PYTHONUTF8=1
python scripts/run_cutdown.py --train data/train.csv --test data/test.csv --profile conservative --cv-folds 3
```

Résultat : `artifacts/submissions/cutdown_conservative.csv` — prêt à soumettre.

---

*Temps total estimé : 5 à 30 minutes selon la complexité du dataset.*
