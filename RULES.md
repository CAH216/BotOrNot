# RÈGLES DE DÉVELOPPEMENT — BotOrNot

> Ces règles s'appliquent à **toutes les contributions** sans exception.
> Aucune mission ne peut les ignorer.

---

## 1. Préserver le pipeline stable

- **Jamais** remplacer `golden_baseline` ou `run_cutdown` sans benchmark comparatif validé
- Toute nouvelle feature ou modèle est ajouté **à côté**, pas à la place
- Le pipeline existant doit continuer à passer tous les tests : `python -m pytest tests/`

## 2. Toute amélioration est derrière un flag ou une config

```yaml
# Exemple dans configs/features.yaml
new_feature_xyz:
  enabled: false   # désactivé par défaut jusqu'à validation
```

```python
# Exemple dans le code
if cfg.get("new_feature_xyz", {}).get("enabled", False):
    X = add_xyz_features(X)
```

## 3. Toute amélioration doit produire une comparaison avant/après

Avant de merger une feature dans le pipeline stable, elle doit produire un rapport avec :

| Métrique | Avant | Après | Delta |
|---|---|---|---|
| AUROC | — | — | — |
| PR-AUC | — | — | — |
| F1 | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| FP | — | — | — |
| FN | — | — | — |
| Temps exec (s) | — | — | — |

Utiliser `scripts/ablation.py` ou `scripts/benchmark.py` pour produire ce rapport.

## 4. Règle de désactivation par défaut

Si l'une de ces conditions est vraie, la feature reste **`enabled: false`** :

- Le gain AUROC est < 0.01 sur les données de validation
- Les résultats sont instables entre runs (σ > 0.02)
- Le nombre de FP augmente sans gain de recall suffisant
- Le temps d'exécution est multiplié par > 2x sans justification

## 5. Ne jamais casser les baselines

Les deux baselines suivantes doivent toujours fonctionner correctement :

```bash
# Golden baseline
python scripts/run_baseline.py --train data/train.csv

# Cut-down baseline (urgence)
python scripts/run_cutdown.py --train data/train.csv --profile conservative
```

Et tous les tests doivent passer :
```bash
python -m pytest tests/ -q
```

---

*Ce fichier fait autorité. En cas de doute, la stabilité prime sur le gain marginal.*
