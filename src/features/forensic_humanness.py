# -*- coding: utf-8 -*-
"""
src/features/forensic_humanness.py
====================================
Forensic Humanness Lab — 6 dimensions contrastives.

Ces features cherchent à détecter la PERFORMANCE D'HUMAIN par un bot :
un bot qui joue l'humain laisse des traces forensiques que de vraies features
temporelles ne voient pas.

Features :
  1. bio_cliche_score        : bio ressemble à un template générique (bot tell)
  2. post_json_artifact      : posts avec artefacts JSON/code (LaLM artifact)
  3. follower_bait_ratio     : follower farming dans les posts
  4. persona_bio_drift       : décalage entre thème de bio et contenus des posts
  5. over_narration_score    : auto-narration explicite ("Just went to the gym")
  6. typo_authenticity       : vrai humain → typos authentiques vs. bot → trop propre OU faux typos
  7. forensic_bot_score      : score composite (plus haut = plus suspect forensiquement)
"""

import re, math
import numpy as np
import pandas as pd
from collections import Counter


# ── Compilations ──────────────────────────────────────────────────────────────

# Bio clichés : patterns génériques de bio LLM/template
_BIO_CLICHE_RE = re.compile(
    r'(\b\w+\s+enthusiast\b'
    r'|\b\w+\s+lover\b'
    r'|\blife\s+is\s+a\s+\w+'
    r'|\bfinding\s+\w+\s+in\s+\w+'
    r'|\beveryday\s+life\b'
    r'|\bnew\s+page\b'
    r'|\bpassionate\s+(about|writer|reader)\b'
    r'|\bprofessional\s+\w+\s+taker\b'
    r'|\bvibes?\b.{0,20}\blover\b'
    r'|\bcoffee\s+(lover|addict)\b'
    r'|\bjust\s+living\s+(my|the)\s+best\b'
    r'|\bdream\s+big\b'
    r'|\bkeeping\s+(it|the)\s+real\b'
    r'|\bmaking\s+(memories|the\s+most)\b)',
    re.I
)

# Artefacts JSON/code dans les posts (LaLM artifacts)
_JSON_RE = re.compile(r'^\s*[\[\{].*[\]\}]\s*$|"[^"]+"\s*:', re.M)

# Follower farming
_FOLLOW_BAIT_RE = re.compile(
    r'\b(welcoming\s+new\s+followers|follow\s+back|gain\s+followers|'
    r'follow\s+for\s+follow|f4f\b|follow\s+me\s+back|new\s+followers\s+right\s+now|'
    r'check\s+out\s+my\s+profile|visit\s+my\s+profile)\b',
    re.I
)

# Sur-narration : descriptions d'activités quotidiennes explicites
_NARRATION_RE = re.compile(
    r'\b(just\s+(went|got|had|finished|came|did|realized)|'
    r'today\s+i\s+(went|had|did|realized|found)|'
    r'nothing\s+beats\s+the\s+feeling\s+of|'
    r'can\'t\s+wait\s+to|'
    r'sipping\s+on|catching\s+up\s+on|'
    r'what\'s\s+on\s+your\s+playlist|'
    r'share\s+your\s+favorites)\b',
    re.I
)

# Typos authentiques humains (pas de mots standard mal orthographiés)
_AUTHENTIC_TYPO_RE = re.compile(
    r'\b(gonna|wanna|gotta|kinda|sorta|idk|imma|tbh|ngl|smh|lmao|lmfao|wtf|omg|imo|iirc|afaik|brb|irl|imo|rn\b|tho\b|bc\b|cuz\b|ur\b|u\b|r\b|np\b|thx\b)',
    re.I
)

# Themes bio pour drift calculation
_THEME_LABELS = {
    'sports':   re.compile(r'\b(sport|football|soccer|basketball|nba|nfl|mlb|nhl|score|game|player|goat|league)\b', re.I),
    'food':     re.compile(r'\b(food|cook|chef|eat|recipe|restaurant|coffee|tea|cheese)\b', re.I),
    'tech':     re.compile(r'\b(tech|developer|code|software|ai|ml|data|programming)\b', re.I),
    'music':    re.compile(r'\b(music|song|album|band|artist|concert|playlist|kpop)\b', re.I),
    'book':     re.compile(r'\b(book|read|write|author|novel|story|poetry|writer)\b', re.I),
    'travel':   re.compile(r'\b(travel|trip|adventure|explore|country|city|tourist)\b', re.I),
    'nature':   re.compile(r'\b(nature|outdoors|hiking|mountain|sea|beach|garden)\b', re.I),
}


# ── Feature Functions ─────────────────────────────────────────────────────────

def _bio_cliche(bio: str) -> float:
    """Score de cliché de bio [0-1]. Plus haut = plus suspect."""
    if not bio or len(bio) < 5:
        return 0.0
    hits = len(_BIO_CLICHE_RE.findall(bio))
    # Normaliser par la longueur de bio
    cliche_density = min(hits / max(len(bio.split()), 1) * 3.0, 1.0)
    return float(np.clip(cliche_density, 0, 1))


def _json_artifact(texts: list) -> float:
    """Ratio de posts avec artefacts JSON [0-1]."""
    if not texts: return 0.0
    hits = sum(1 for t in texts if _JSON_RE.search(t))
    return round(hits / len(texts), 4)


def _follower_bait(texts: list) -> float:
    """Ratio de posts de follower farming [0-1]."""
    if not texts: return 0.0
    hits = sum(1 for t in texts if _FOLLOW_BAIT_RE.search(t))
    return round(min(hits / len(texts) * 5.0, 1.0), 4)  # Amplifié ×5


def _persona_drift(bio: str, texts: list) -> float:
    """
    Décalage thématique entre la bio et les posts.
    Un bot performance souvent une bio de sportif mais publie du contenu dispersé.
    Retourne 0 si cohérent, 1 si totalement incohérent.
    """
    if not bio or not texts:
        return 0.5  # inconnu → neutre

    # Détecter le thème dominant de la bio
    bio_themes = {t: bool(pat.search(bio)) for t, pat in _THEME_LABELS.items()}
    main_bio_theme = [t for t, v in bio_themes.items() if v]

    if not main_bio_theme:
        return 0.0  # Pas de thème clair dans la bio → pas de drift mesurable

    main_theme = main_bio_theme[0]
    theme_pat = _THEME_LABELS[main_theme]

    # Compter les posts qui correspondent au thème de la bio
    on_theme = sum(1 for t in texts if theme_pat.search(t))
    on_theme_ratio = on_theme / len(texts)

    # Si la bio annonce un thème mais peu de posts sont dans ce thème → drift élevé
    drift = 1.0 - on_theme_ratio
    return round(float(np.clip(drift, 0, 1)), 4)


def _over_narration(texts: list) -> float:
    """Ratio de posts de sur-narration explicite [0-1]."""
    if not texts: return 0.0
    hits = sum(1 for t in texts if _NARRATION_RE.search(t))
    return round(hits / len(texts), 4)


def _typo_authenticity(texts: list) -> float:
    """
    Score d'authenticité des typos [0-1].
    Un vrai humain utilise des abréviations authentiques (gonna, tbh, ngl...).
    Un bot peut être parfait (0 =suspect) ou utiliser de faux mots cool.
    Retourne la densité d'argot authentique — inverse du score de suspicion.
    """
    if not texts: return 0.0
    all_text = " ".join(texts)
    hits = len(_AUTHENTIC_TYPO_RE.findall(all_text))
    word_count = max(len(all_text.split()), 1)
    # 0.05-0.15 = typique humain; < 0.02 = suspect (trop propre)
    argot_ratio = hits / word_count
    # Convertir en score de SUSPICION (manque d'authenticité)
    # Si < 0.02 → suspect (trop clean), score = 1
    # Si > 0.05 → humain → score = 0
    suspicion = max(0, 1 - argot_ratio / 0.04)
    return round(float(np.clip(suspicion, 0, 1)), 4)


# ── Extracteur Principal ───────────────────────────────────────────────────────

def extract_forensic_humanness(
    u_df: pd.DataFrame,
    p_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retourne {user_id, forensic_*} — 7 features de détection forensique.
    Plus le `forensic_bot_score` est élevé, plus le compte ressemble
    à un bot qui PERFORME l'humain.
    """
    if u_df.empty:
        return pd.DataFrame()

    uid_col = "user_id"
    posts_by_uid = {str(uid): [] for uid in u_df[uid_col].values}
    if not p_df.empty and "text" in p_df.columns:
        p_txt = p_df[[uid_col, "text"]].copy()
        p_txt["text"] = p_txt["text"].fillna("").astype(str)
        for _, row in p_txt.iterrows():
            uid = str(row[uid_col])
            if uid in posts_by_uid:
                posts_by_uid[uid].append(row["text"])

    records = []
    for _, u_row in u_df.iterrows():
        uid   = str(u_row[uid_col])
        bio   = str(u_row.get("description", "") or "")
        texts = posts_by_uid.get(uid, [])

        cliche   = _bio_cliche(bio)
        json_art = _json_artifact(texts)
        follow_b = _follower_bait(texts)
        drift    = _persona_drift(bio, texts)
        narrate  = _over_narration(texts)
        typo_sus = _typo_authenticity(texts)

        # Composite : weighted sum — signals les plus discriminants en premier
        forensic = (
            json_art   * 0.30 +   # JSON artifacts = quasi-unique aux bots LLM
            follow_b   * 0.25 +   # Follower bait = signal bot fort
            cliche     * 0.20 +   # Bio cliché = signal modéré
            drift      * 0.10 +   # Persona drift = signal faible
            narrate    * 0.10 +   # Sur-narration = signal faible
            typo_sus   * 0.05     # Manque de typos = signal très faible
        )

        records.append({
            uid_col:                   uid,
            "forensic_bio_cliche":     round(cliche,   3),
            "forensic_json_artifact":  round(json_art, 4),
            "forensic_follow_bait":    round(follow_b, 3),
            "forensic_persona_drift":  round(drift,    3),
            "forensic_over_narration": round(narrate,  3),
            "forensic_typo_suspicion": round(typo_sus, 3),
            "forensic_bot_score":      round(forensic, 4),
        })

    return pd.DataFrame(records)
