# Rapport Comparatif des Évènements 30 (EN) et 31 (FR)

## Event30_English

| Profil | AUROC | PR-AUC | F1-Score | Precision | Recall | Faux Positifs | Faux Négatifs |
|--------|-------|--------|----------|-----------|--------|---------------|---------------|
| **conservative** | 0.993 | 0.976 | 0.887 | 0.948 | 0.833 | 3 | 11 |
| **balanced** | 0.994 | 0.983 | 0.905 | 0.950 | 0.864 | 3 | 9 |
| **aggressive** | 0.994 | 0.983 | 0.922 | 0.952 | 0.894 | 3 | 7 |

## Event31_French

| Profil | AUROC | PR-AUC | F1-Score | Precision | Recall | Faux Positifs | Faux Négatifs |
|--------|-------|--------|----------|-----------|--------|---------------|---------------|
| **conservative** | 0.980 | 0.901 | 0.776 | 0.864 | 0.704 | 3 | 8 |
| **balanced** | 0.981 | 0.912 | 0.769 | 0.800 | 0.741 | 5 | 7 |
| **aggressive** | 0.981 | 0.912 | 0.800 | 0.786 | 0.815 | 6 | 5 |

## 💡 Analyse & Retours: Anglais vs Français
L'évènement francophone montre généralement une robustesse linguistique différente due à la rareté textuelle (Event 31).
L'Application du profil `conservative` y réduit considérablement la casse sur les Faux Positifs (Humains flaggés par erreur).
