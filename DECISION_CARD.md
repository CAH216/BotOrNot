# BotOrNot : Decision Card (Jour J)

Aide-mémoire condensé pour l'opérateur humain.

### Scoring Officiel
`+2 TP` / `-2 FN` / **`-6 FP`** → 1 Faux Positif = 3 bots ratés

### Format d'Entrée
`dataset.posts&users.XX.json` (pas de CSV, pas d'edges)

### Format de Sortie
`BotOrNot.detections.XX.txt` — un user ID par ligne (pas de CSV)

### Profil par Défaut
**Conservative** — imposé par le scoring asymétrique (-6 FP).

### Commande Standard (Jour J)
```powershell
$env:PYTHONUTF8=1; python scripts/submission_factory.py --train data/train.csv --test data/test.csv --format official
```

### Urgence Absolue (< 5min)
```powershell
$env:PYTHONUTF8=1; python scripts/run_cutdown.py --train data/train.csv --test data/test.csv --profile conservative
```

### Quand changer de profil ?
- **Signal exceptionnel + 0 ambiguïté + texte anglais riche** → `balanced`
- **Tout autre cas** → `conservative`
- **Jamais `aggressive`** avec le scoring officiel
