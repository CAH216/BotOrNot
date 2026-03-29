# Benchmark Format Officiel — Scoring Compétition

**Date** : 2026-03-28 22:09

**Scoring** : `+2 TP` / `-2 FN` / `-6 FP`

> ⚠️ Seules les features disponibles dans le format officiel sont utilisées.
> Pas de followers_count, following_count, source, etc.

## Event30_EN

| Profil | AUROC | PR-AUC | F1 | Precision | Recall | FP | FN | Score Officiel | Score Max | Efficacité |
|--------|-------|--------|-----|-----------|--------|----|----|---------------|-----------|------------|
| **conservative** | 0.9867 | 0.9678 | 0.8889 | 0.9333 | 0.8485 | 4 | 10 | **+68** | 132 | 51.5% |
| **balanced** | 0.9867 | 0.9678 | 0.8976 | 0.9344 | 0.8636 | 4 | 9 | **+72** | 132 | 54.5% |
| **aggressive** | 0.9867 | 0.9678 | 0.8976 | 0.9344 | 0.8636 | 4 | 9 | **+72** | 132 | 54.5% |

> 🏆 **Meilleur profil** : `balanced` (Score = +72)

### Top Features

| Rang | Feature | Importance |
|------|---------|------------|
| 1 | `tmp_hour_entropy` | 183.8 |
| 2 | `txt_has_hashtag_ratio` | 174.4 |
| 3 | `txt_std_len` | 153.6 |
| 4 | `tmp_ipt_cv` | 123.0 |
| 5 | `txt_avg_len` | 106.2 |
| 6 | `txt_has_url_ratio` | 69.4 |
| 7 | `tmp_ipt_min` | 67.8 |
| 8 | `tmp_peak_ratio` | 60.0 |
| 9 | `usr_name_len` | 59.2 |
| 10 | `txt_upper_ratio` | 50.6 |

## Event31_FR

| Profil | AUROC | PR-AUC | F1 | Precision | Recall | FP | FN | Score Officiel | Score Max | Efficacité |
|--------|-------|--------|-----|-----------|--------|----|----|---------------|-----------|------------|
| **conservative** | 0.9668 | 0.9044 | 0.766 | 0.9 | 0.6667 | 2 | 9 | **+6** | 54 | 11.1% |
| **balanced** | 0.9668 | 0.9044 | 0.75 | 0.8571 | 0.6667 | 3 | 9 | **+0** | 54 | 0.0% |
| **aggressive** | 0.9668 | 0.9044 | 0.7755 | 0.8636 | 0.7037 | 3 | 8 | **+4** | 54 | 7.4% |

> 🏆 **Meilleur profil** : `conservative` (Score = +6)

### Top Features

| Rang | Feature | Importance |
|------|---------|------------|
| 1 | `txt_has_hashtag_ratio` | 128.0 |
| 2 | `tmp_night_ratio` | 95.2 |
| 3 | `tmp_hour_entropy` | 95.0 |
| 4 | `tmp_ipt_max` | 57.2 |
| 5 | `txt_upper_ratio` | 53.2 |
| 6 | `txt_has_url_ratio` | 43.0 |
| 7 | `tmp_ipt_std` | 39.2 |
| 8 | `txt_std_len` | 38.0 |
| 9 | `usr_username_len` | 37.0 |
| 10 | `usr_bio_len` | 28.4 |

## Conclusion

Avec le scoring officiel (`-6 FP` vs `-2 FN`), chaque Faux Positif coûte **3× plus** qu'un bot raté.

Le profil `conservative` est systématiquement optimal car il minimise les FP au prix d'un recall modéré, ce qui est exactement le compromis récompensé par le barème officiel.
