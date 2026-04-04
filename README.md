# 🤖 BotOrNot — The Ultimate Forensic Detection Pipeline 🏆

Le Système BotOrNot le plus inviolable et abouti jamais produit pour la détection comportementale inorganique.

Optimisé pour le **Scoring Officiel Hétérogène** :  `+2 TP` / `-2 FN` / **`-6 FP`**.
Cette architecture est structurée selon l'approche **"Multi-Layered Pareto Guardrails"**.

---

## ⚡ Architecture Globale (Tri-Core System)

### 1. Le Monolithe (LightGBM Base)
Algorithme de Machine Learning optimisé pour la pré-détection de haute précision.
- Extrait des features mathématiques croisées depuis `src/champion_config.py`.
- Effectue un Cross-Validation K-Fold Stratifié pour éviter tout overfitting spatial.
- Scrypte le Nuage de points initial et écarte massivement la "Preuve Simple".

### 2. Forensic K-NN Court (Le Tribunal Analytique)
Un système multi-bureau de proximité vectorielle reposant sur le `CandidateMinerCourt`.
- **Topologie K-NN Pairwise** : Vérifie l'ensemble des cas dans les sous-espaces (*Stylométrique*, *Temporel*).
- **Dual Court (Atypical Human Rescue Bank)** : Isole dynamiquement les Faux Positifs structurels de l'entraînement pour bâtir une jauge de sauvegarde de l'Humain Bizarre (Atypical Human). Aucune fuite de données, 100% autogénéré.

### 3. Final Arbitration Judge (Llama-3.3 LLM Gatekeeper)
Le bouclier ultime de la cybersécurité. Appelé exclusivement en cas de litige sévère entre le Modèle ML et les Tribunaux d'incohérence, ce juge a pour instruction de trancher les **Faux Positifs Résiduels**.
- Connecté à l'API **Groq**.
- Fonctionne avec des Seuils de Veto (Micro-Veto) inébranlables fixés en Python (`confidence >= 0.80`).
- Parfaitement inactif quand indésiré, et ultra performant quand convoqué pour de l'extraction chirurgicale de faux virtuels (Statut : *KEEP_EXPERIMENTAL / PASSIVE SHIELD*).

---

## 🚀 Commande de Soumission Héroïque

```powershell
# Déploiement Officiel du Concours
$env:PYTHONUTF8=1; python scripts/run_appeal_eval.py (ou équivalent métier fourni pour l'Event final)
```

*(Note : Toutes les prédictions passent inévitablement par les 3 filtres : `Base prob > K-NN Adjudication > Python Hard-Rule Arbitration`).*

---

## 📁 Structure du Projet

```
BotOrNot/
│
├── configs/
├── data/                        # Datasets (Ignorés dans GitHub)
│
├── src/
│   ├── champion_config.py              # Le cœur des Hyper-Paramètres & Synthétisation
│   ├── features/
│   │   ├── candidate_miner_court.py    # Moteur Topologique Dual Court
│   │   ├── groq_general_judge.py       # Le Juge LLM Principal System
│   │   ├── final_arbitration_judge.py  # Le Verrou Mathématique Python sur l'IA
│   │   ├── forensic_humanness.py       # Analyseurs de signatures Humaines
│   │   └── ghost_human_protector_v2.py # GHP Bouclier de Volume
│
├── scripts/
│   ├── residual_error_autopsy.py       # Extracteur de faille
│   └── run_dual_court_benchmark.py     # Ultimate Benchmark Validateur
│
└── tests/
```

🛠️ *Cette configuration a été benchmarkée sur l'Event 5 et 6 et démontre une robustesse à 0 dégradation, figeant le TP/FN/FP optimal.*
