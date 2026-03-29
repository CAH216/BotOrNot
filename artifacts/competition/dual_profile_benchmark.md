# Benchmark Dual Profiles EN/FR

**Date** : 2026-03-28 22:44

**Scoring** : `+2 TP` / `-2 FN` / `-6 FP`

## Event30_EN

| Profil | Seuil | Score | TP | FP | FN | Precision | Recall |
|--------|-------|-------|----|----|----|-----------|--------|
| Conservative (t=0.60) | 0.60 | **+68** | 56 | 4 | 10 | 0.933 | 0.849 |
| EN-Optimized (t=0.30) **◄** | 0.30 | **+76** | 58 | 4 | 8 | 0.935 | 0.879 |
| FR-Optimized (t=0.60) | 0.60 | **+68** | 56 | 4 | 10 | 0.933 | 0.849 |
| Balanced (t=0.50) | 0.50 | **+72** | 57 | 4 | 9 | 0.934 | 0.864 |
| Aggressive (t=0.38) | 0.38 | **+72** | 57 | 4 | 9 | 0.934 | 0.864 |

## Event31_FR

| Profil | Seuil | Score | TP | FP | FN | Precision | Recall |
|--------|-------|-------|----|----|----|-----------|--------|
| Conservative (t=0.60) | 0.60 | **+6** | 18 | 2 | 9 | 0.900 | 0.667 |
| EN-Optimized (t=0.30) | 0.30 | **+8** | 20 | 3 | 7 | 0.870 | 0.741 |
| FR-Optimized (t=0.60) **◄** | 0.60 | **+6** | 18 | 2 | 9 | 0.900 | 0.667 |
| Balanced (t=0.50) | 0.50 | **+0** | 18 | 3 | 9 | 0.857 | 0.667 |
| Aggressive (t=0.38) | 0.38 | **+4** | 19 | 3 | 8 | 0.864 | 0.704 |

## Gain Combiné

| Stratégie | Event 30 (EN) | Event 31 (FR) | **Total** |
|-----------|--------------|--------------|----------|
| Conservative unifié (t=0.60) | +68 | +6 | **+74** |
| **Dual EN/FR** | +76 | +6 | **+82** |

> **Delta : +8 points — GAIN VALIDÉ**
