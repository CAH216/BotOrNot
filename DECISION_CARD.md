# ⚡ BOTORNOT — DECISION CARD (DAY J)

| Contexte / Signal | 🟢 Action Immédiate |
|---|---|
| **Métrique inconnue (Kaggle)** | Ciblez `F1-Score` et soumettez `submission_balanced.csv`. |
| **Format `proba_only` requis** | Ciblez `AUROC` et soumettez `submission_aggressive.csv` (proba brute, 0 filtre). |
| **Timestamps très riches**| Activez `temporal` et `coordination` dans `configs/features.yaml`. |
| **Texte très faible (absent)**| Coupez `text_model`, gardez seulement `text_basic`. |
| **Urgence absolue (Temps < 5min)**| `$env:PYTHONUTF8=1; python scripts/submission_factory.py --train data/train.csv --test data/test.csv` |
| **Urgence extrême (Cut-Down)**| `$env:PYTHONUTF8=1; python scripts/run_baseline.py --train data/train.csv --test data/test.csv --profile balanced` |
| **Doute sur la statégie**| `$env:PYTHONUTF8=1; python scripts/meta_ranker.py --train data/train.csv` |

---

### 🎯 PROFIL PAR DÉFAUT (Le Choix en cas de Doute)
- **Défaut absolu si rien n'est clair** → `submission_balanced.csv`
- **Si format proba_only / Maximize AUROC** → `submission_aggressive.csv`
- **Si risque de Faux Positifs élevé** → `submission_conservative.csv`
