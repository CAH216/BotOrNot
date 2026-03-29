# Rapport de Robustesse Adversariale

**Date** : 2026-03-28 16:52

## 1. Comparaison des Performances

| Scénario | AUROC | PR-AUC | F1-Score | Precision | Recall | FP | FN | ΔAUROC | ΔF1 | Risque |
|----------|-------|--------|----------|-----------|--------|----|----|--------|-----|--------|
| **baseline** | 1.0 | 1.0 | 0.9967 | 0.9934 | 1.0 | 1 | 0 | — | — | Réf. |
| **sleeper_bot** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | +0.0000 | +0.0033 | 🟢 SAFE |
| **jitter_bot** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | +0.0000 | +0.0033 | 🟢 SAFE |
| **llm_bot** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | +0.0000 | +0.0033 | 🟢 SAFE |
| **camouflage_bot** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | +0.0000 | +0.0033 | 🟢 SAFE |

## 2. Importance des Familles de Features par Scénario

| Scénario | Temporal | Tabular | Text | Structural | Autre |
|----------|----------|---------|------|------------|-------|
| **baseline** | 2.7% | 97.0% | 0.3% | 0.0% | 0.0% |
| **sleeper_bot** | 38.3% | 60.9% | 0.9% | 0.0% | 0.0% |
| **jitter_bot** | 38.5% | 60.3% | 1.2% | 0.0% | 0.0% |
| **llm_bot** | 38.9% | 60.4% | 0.7% | 0.0% | 0.0% |
| **camouflage_bot** | 38.9% | 60.3% | 0.8% | 0.0% | 0.0% |

## 3. Top 10 Features (Baseline)

| Rang | Feature | Importance |
|------|---------|------------|
| 1 | `raw_followers_count` | 127.0 |
| 2 | `raw_statuses_count` | 63.6 |
| 3 | `raw_following_count` | 59.4 |
| 4 | `tmp_ipt_cv` | 4.8 |
| 5 | `tab_ff_sum` | 3.2 |
| 6 | `tmp_ipt_mean` | 1.8 |
| 7 | `tab_ff_asymmetry` | 1.6 |
| 8 | `txt_len_mean` | 0.8 |
| 9 | `tab_ff_ratio` | 0.6 |
| 10 | `tmp_peak_ratio` | 0.2 |

## 4. Analyse des Vulnérabilités

### Menace la plus dangereuse
- Aucune menace significative détectée.

### Modules qui résistent le mieux
- **sleeper_bot** : temporal ↑35.6%, tabular ↓36.2%
- **jitter_bot** : temporal ↑35.9%, tabular ↓36.8%
- **llm_bot** : temporal ↑36.2%, tabular ↓36.6%
- **camouflage_bot** : temporal ↑36.3%, tabular ↓36.8%

### Recommandation Globale

> 🟢 **Niveau : SAFE** — Le pipeline résiste bien aux scénarios adversariaux testés.
> Aucune action immédiate nécessaire.
