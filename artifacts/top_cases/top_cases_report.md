# Rapport de Diagnostic — Top Cas (Mission 11)
**Date** : 2026-03-27 23:26:04
**Seuil de Décision** : 0.5
**Filtre Anti-FP** : `balanced`

## 1. Les 20 Bots les plus évidents (Vrais Positifs Forts)
*Les comptes bots que le modèle détecte avec la plus forte certitude.*

| Account ID | Raw Prob | Adj Prob | Anti-FP | Modules → BOT | Modules → HUMAIN |
|---|---|---|---|---|---|
| `u0` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u49` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u50` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u51` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u52` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u53` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u54` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u55` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u56` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u57` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u58` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u59` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u6` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u60` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u61` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u62` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u63` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u5` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u48` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |
| `u65` | 0.998 | **0.998** | Non | tabular(1.00), temporal(1.00), text_basic(1.00) | Aucun |


## 2. Les 20 Humains héroïquement sauvés (True Negatives protégés)
*Les comptes humains que le modèle trouvait suspects (proba proche ou > seuil) mais que le filtre Anti-Faux-Positifs a pénalisés à juste titre.*

Aucun cas dans cette catégorie.

## 3. Les 20 Faux Positifs tenaces
*Les humains que le pipeline continue de bannir (Erreur critique). Observez les modules qui trompent le score.*

Aucun cas dans cette catégorie.

## 4. Les 20 Faux Négatifs indétectables
*Les bots qui passent totalement sous le radar (vus comme très humains). Observez l'absence de signaux.*

Aucun cas dans cette catégorie.
