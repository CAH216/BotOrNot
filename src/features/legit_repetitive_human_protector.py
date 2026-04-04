# -*- coding: utf-8 -*-
"""
src/features/legit_repetitive_human_protector.py
=================================================
Protège les humains atypiques légitimes qui ressemblent à des bots
parce que leur comportement est répétitif par INTENTION et non par MÉCANIQUE.

Profils ciblés (confirmés sur Event 5 FP) :
  - Poll accounts (@no_context_poll)     → répétitif mais questions variées, engagement
  - Stan/Fandom accounts (@sapalocha98)  → 100% hashtags mais culture communautaire
  - Quote/RP accounts (@rise_quotes)     → bio "bot/quote" mais persona fictive cohérente
  - Sports livetweet (@barcalover_100)   → peu de nuit mais fan actif diurne
  - Music promo humans (@MusicMaven123)  → promo mais activité propre à un humain

Principe : produire un score de LÉGITIMITÉ RÉPÉTITIVE (pas un décideur).
  0.0 = comportement suspicieux / mécanique pur
  1.0 = fortement marqué "humain intentionnel"
"""

import re
import math
import numpy as np
import pandas as pd
from collections import Counter

# ─── Constantes ───────────────────────────────────────────────────────────────

# Mots de fandom / stan culture
_FANDOM_WORDS = {
    "stan", "stans", "stanning", "fandom", "fanbase", "ot7", "ult", "bias",
    "kpop", "blinks", "army", "once", "exol", "carat", "stay", "moa",
    "directioner", "mixer", "swiftie", "beyhive", "barb", "monster",
    "skincare", "shipper", "girlgroup", "boygroup", "maknae", "unnie",
    "senpai", "weeb", "otaku", "ship", "ult bias", "pfp", "comeback",
    "stream", "mv", "album", "photocard", "lightstick",
}

# Indicateurs de comptes quote / RP / parody
_RP_BIO_PATTERNS = [
    r"\bquote\b", r"\bquotes\b", r"\brp\b", r"\broleplay\b",
    r"\bparody\b", r"\bfan account\b", r"\bfan acc\b", r"\bnot affiliated\b",
    r"\bfictional\b", r"\bcharacter\b", r"\bvoice\b", r"\bsatire\b",
    r"\bbot\b",  # "quote bot", "post bot", etc.
    r"\btaken from\b", r"\bquoted from\b", r"\blines from\b",
]

# Indicateurs de livetweet / communauté sportive
_SPORTS_PATTERNS = [
    r"\b(scores?|goals?|points?|rebounds?|assists?|shots?|fouls?)\b",
    r"\b(q[1-4]|half|OT|overtime|halftime|quarterfinal|final)\b",
    r"\b(win|wins|loss|beat|beats|defeated|scored|led|trailed)\b",
    r"\b(nba|nfl|mlb|nhl|nba|ucl|premier|serie a|laliga|bundesliga|ligue 1)\b",
    r"\b(lakers|warriors|celtics|nets|knicks|heat|bulls|cavs|bucks|suns)\b",
    r"\b(touchdown|slam dunk|hat trick|home run|ace|birdie|par|bogey)\b",
]

# Mots de promotion légitime (promo d'un vrai service, pas spam pur)
_LEGIT_PROMO_MARKERS = [
    r"\bfree trial\b", r"\bmusic submission\b", r"\bstreams?\b",
    r"\bpromotion\b", r"\bsubmit\b", r"\bplaylist\b", r"\bartist\b",
    r"\bfollowers?\b", r"\byoutube\b", r"\binstagram\b", r"\bspotify\b",
]

# Patterns de poll / question engagement
_POLL_PATTERNS = [
    r"\?$",                         # Ligne finissant par ?
    r"^(who|what|when|where|why|how|would|could|should|is|are|do|did|can)\b",
    r"\b(top\s*\d+|favorite|favourite|best|worst|rank)\b",
    r"\b(vs\.?|versus|or)\b",       # A vs B comparisons
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _any_match(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)


def _fandom_score(texts: list, bio: str) -> float:
    """Détecte la culture stan/fandom dans bio + posts."""
    words = set()
    for t in texts:
        words.update(re.findall(r'\b\w+\b', str(t).lower()))
    bio_words = set(re.findall(r'\b\w+\b', str(bio).lower()))
    all_words = words | bio_words
    overlap = len(all_words & _FANDOM_WORDS)
    return min(overlap / 3.0, 1.0)  # 3+ mots fandom = score 1.0


def _rp_quote_score(bio: str, texts: list) -> float:
    """Détecte les comptes RP/Quote/Parody via la bio et la cohérence des posts."""
    bio_score = 1.0 if _any_match(bio, _RP_BIO_PATTERNS) else 0.0
    
    # Les posts RP sont cohérents (même univers) mais sans spam de liens
    if not texts:
        return bio_score * 0.5
    
    url_ratio = sum(1 for t in texts if "http" in str(t)) / max(len(texts), 1)
    # Un vrai RP ne spam pas de liens externes
    no_link_boost = 0.3 if url_ratio < 0.2 else 0.0
    
    return min(bio_score + no_link_boost, 1.0)


def _poll_engagement_score(texts: list) -> float:
    """Détecte les comptes de sondages et d'engagement communautaire."""
    if not texts:
        return 0.0
    
    question_posts = sum(1 for t in texts if _any_match(str(t), _POLL_PATTERNS))
    question_ratio = question_posts / max(len(texts), 1)
    
    # Diversité des sujets des hashtags (un poll humain couvre plusieurs topics)
    all_tags = []
    for t in texts:
        all_tags.extend(re.findall(r'#(\w+)', str(t).lower()))
    
    tag_diversity = 0.0
    if len(all_tags) >= 2:
        unique_tags = len(set(all_tags))
        tag_diversity = min(unique_tags / max(len(all_tags), 1), 1.0)
    
    return min(question_ratio * 0.6 + tag_diversity * 0.4, 1.0)


def _sports_livetweet_score(texts: list, bio: str, night_ratio: float) -> float:
    """Détecte les fans sportifs / live-tweeters diurnes."""
    if not texts:
        return 0.0
    
    sport_posts = sum(1 for t in texts if _any_match(str(t), _SPORTS_PATTERNS))
    sport_ratio = sport_posts / max(len(texts), 1)
    
    bio_sport = 1.0 if _any_match(bio, _SPORTS_PATTERNS + [r"\bfan\b", r"\bfootball\b", r"\bbasketball\b"]) else 0.0
    
    # Un fan sportif peut avoir peu de posts nocturnes (jeux en journée/soirée)
    diurnal_legit = 0.3 if night_ratio < 0.15 else 0.0
    
    return min(sport_ratio * 0.5 + bio_sport * 0.3 + diurnal_legit, 1.0)


def _promo_service_human_score(texts: list, bio: str) -> float:
    """
    Distingue le promoteur humain légitime du spam pur.
    Un humain de promo a du contenu mixte (promo + personnel).
    Un bot spam n'a QUE de la promo identique.
    """
    if not texts:
        return 0.0
    
    is_promo_bio = 1.0 if _any_match(bio, _LEGIT_PROMO_MARKERS) else 0.0
    if is_promo_bio < 0.5:
        return 0.0  # Pas un compte promo, ne pas scorer
    
    promo_posts = sum(1 for t in texts if _any_match(str(t), _LEGIT_PROMO_MARKERS))
    promo_ratio = promo_posts / max(len(texts), 1)
    
    # Variété de longueur = humain (les bots spam ont des longueurs stables)
    lengths = [len(str(t)) for t in texts]
    len_cv = np.std(lengths) / (np.mean(lengths) + 1e-9) if len(lengths) > 1 else 0.0
    variety_bonus = min(len_cv * 0.5, 0.4)
    
    # Partiel = entre 30-80% promo est typique d'un humain qui gère sa promo
    if 0.3 <= promo_ratio <= 0.85:
        return min(0.6 + variety_bonus, 1.0)
    return variety_bonus  # Promo totale → bonus faible seulement


def _intentionality_score(texts: list) -> float:
    """
    Mesure l'intentionnalité humaine : variation d'émotion, auto-référence, humour, langue familière.
    Un bot mécanique est mono-ton même avec des templates variés.
    """
    if not texts:
        return 0.0
    
    has_self_ref  = sum(1 for t in texts if re.search(r"\b(i|me|my|mine|we|us|our)\b", str(t), re.I))
    has_humor     = sum(1 for t in texts if re.search(r"\b(lol|lmao|haha|hehe|omg|wtf|bruh|ngl|smh|💀|😂|🤣)\b", str(t), re.I))
    has_reaction  = sum(1 for t in texts if re.search(r"\b(wait|omg|honestly|literally|actually|nope|yep|nah|yikes)\b", str(t), re.I))
    has_question_to_others = sum(1 for t in texts if re.search(r"@\w+.*\?|^anyone|^does anyone", str(t), re.I))
    
    n = max(len(texts), 1)
    intentional = (
        (has_self_ref / n) * 0.25 +
        (has_humor / n) * 0.25 +
        (has_reaction / n) * 0.25 +
        (has_question_to_others / n) * 0.25
    )
    return min(intentional, 1.0)


# ─── Extracteur Principal ──────────────────────────────────────────────────────

def extract_legit_repetitive_human_protector(
    u_df: pd.DataFrame,
    p_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retourne un DataFrame {user_id, lrh_*} avec le score de légitimité répétitive.
    Toutes les valeurs sont dans [0, 1].
    """
    if u_df.empty:
        return pd.DataFrame()

    uid_col = "user_id"
    records = []

    # Pré-calcul des temporels
    night_by_uid = {}
    if "created_at" in p_df.columns and not p_df.empty:
        p_work = p_df[[uid_col, "created_at"]].copy()
        p_work["_ts"] = pd.to_datetime(p_work["created_at"], utc=True, errors="coerce")
        p_work["_hour"] = p_work["_ts"].dt.hour
        for uid, grp in p_work.groupby(uid_col):
            hours = grp["_hour"].dropna()
            night_by_uid[str(uid)] = float(hours.isin(range(0, 6)).mean()) if len(hours) > 0 else 0.25

    # Grouper les posts par uid
    posts_by_uid: dict = {str(uid): [] for uid in u_df[uid_col].values}
    if not p_df.empty and "text" in p_df.columns:
        p_txt = p_df[[uid_col, "text"]].copy()
        p_txt["text"] = p_txt["text"].fillna("").astype(str)
        for _, row in p_txt.iterrows():
            uid = str(row[uid_col])
            if uid in posts_by_uid:
                posts_by_uid[uid].append(row["text"])

    for _, u_row in u_df.iterrows():
        uid  = str(u_row[uid_col])
        bio  = str(u_row.get("description", "") or "")
        texts = posts_by_uid.get(uid, [])
        night = night_by_uid.get(uid, 0.25)

        # Calcul des 6 sub-scores
        sc_fandom   = _fandom_score(texts, bio)
        sc_rp       = _rp_quote_score(bio, texts)
        sc_poll     = _poll_engagement_score(texts)
        sc_sports   = _sports_livetweet_score(texts, bio, night)
        sc_promo    = _promo_service_human_score(texts, bio)
        sc_intent   = _intentionality_score(texts)

        # Score composite de légitimité : max-fusion pondérée
        # Un seul signal fort suffit pour protéger
        type_scores = [sc_fandom, sc_rp, sc_poll, sc_sports, sc_promo]
        specialist_max = max(type_scores)

        # Combinaison : max-type * 0.6 + intentionnalité * 0.4
        legit_score = specialist_max * 0.6 + sc_intent * 0.4

        records.append({
            uid_col:              uid,
            "lrh_fandom_score":   round(sc_fandom,  3),
            "lrh_rp_score":       round(sc_rp,      3),
            "lrh_poll_score":     round(sc_poll,     3),
            "lrh_sports_score":   round(sc_sports,   3),
            "lrh_promo_score":    round(sc_promo,    3),
            "lrh_intent_score":   round(sc_intent,   3),
            "lrh_specialist_max": round(specialist_max, 3),
            "lrh_legit_score":    round(legit_score, 3),
        })

    res = pd.DataFrame(records)
    return res
