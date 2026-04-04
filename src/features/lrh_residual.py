# -*- coding: utf-8 -*-
"""
src/features/lrh_residual.py
============================
LRH Residual — Protecteur des 3 archétypes FP résistants.

Ciblés sur les signatures exactes diagnostiquées (Event 5) :

1. POLL HUMAN (@no_context_poll)
   Signal distinctif : posts à structure de question (même sans "?" final),
   haute diversité de hashtags topicaux (127 unique / 363 total),
   aucun CTA spam, bio engageante personnelle.

2. QUOTE / RP / PARODY HUMAN (@rise_quotes)
   Signal distinctif : bio explicite "bot/quote/taken from",
   posts sans URL, cohérence thématique d'un seul univers fictif,
   style uniforme mais non-promotionnel.

3. STAN / FANDOM HUMAN (@sapalocha98)
   Signal distinctif : alternance blagues/stan-promo en pattern fixe,
   23+ posts de jeux de mots (puns), très peu de diversité de hashtags
   (ultra-fandom = 2 unique tags), posts personnels mélangés.

Principe : score protecteur uniquement [0, 1].
Aucune logique d'accusation. Feature injectée dans LightGBM.
"""

import re
import math
import numpy as np
import pandas as pd
from collections import Counter

# ─── Helpers ──────────────────────────────────────────────────────────────────

_URL_RE  = re.compile(r'https?://\S+|www\.\S+', re.I)
_CTA_RE  = re.compile(
    r'\b(click|link in bio|dm me|buy now|get yours|promo|discount|coupon|limited offer|order now)\b',
    re.I
)
_QUESTION_MARKERS = re.compile(
    r'\b(who|what|when|where|why|how|would|could|should|is it|are you|do you|did you|can you|which|rank|best|worst|top\s*\d+|vs\.?|versus|favorite|favourite|greatest|ever|all time)\b',
    re.I
)
_PUN_SETUP = re.compile(
    r'\b(why did(n.t)?|what did|what do you call|what happens when|knock knock|did you hear|how do you|what goes|why does|what.s the difference|what.s a)\b',
    re.I
)
_RP_BIO_RE = re.compile(
    # PATTERNS STRICTS : exige une déclaration de CONTENU fictif, pas juste une négation
    # EXCLU : 'not affiliated' (disclaimer courant chez les bots), 'bot' seul
    r'\b(quotes?\s+(?:taken|from|by)|taken\s+from|lines\s+from|quoting\s+@|'
    r'roleplay|rp\s+account|fan\s+acc(?:ount)?|parody\s+(?:account|of)|'
    r'character\s+bot|post\s+bot|(?:quote|fan)\s+bot)\b',
    re.I
)
_STAN_MARKERS = {
    'stan','stans','stanning','ult','bias','comeback','lightstick','stream',
    'mv','blink','army','once','stay','moa','carat','orbit','dive','reveluv',
    'engene','midzy','ahgase','aroha','army','exol','inc','fandom','stan twitter',
    'kpop','kpopper','girlgroup','boygroup','photo card','pc','weverse','vlive',
}


# ─── Archétype 1 : Poll Human ─────────────────────────────────────────────────

_POLL_EXPLICIT = re.compile(
    r'(top\s*\d+|#\d+\s+vs\.?|who would win|pick one|your fav(?:ourite)?|rate|rank|poll|would you rather|\bA\)\s|\bB\)\s|1\.\s+\w|\bIs it\b|\bOr\b.*\?)',
    re.I
)

def _poll_human_v2(texts: list, bio: str) -> float:
    """
    Poll human : compte de sondages ET d'engagement.
    Doit avoir un FORMAT explicite de poll (Top N, vs, liste A/B, question d'opinion)
    ET une grande diversité de hashtags (≥ 20 total avec diversity ≥ 0.25).
    Guard : si les posts ont peu d'hashtags totaux c'est un bot narratif.
    """
    if not texts:
        return 0.0

    n = len(texts)

    # 1. Format de poll EXPLICITE (pas juste un mot interrogatif)
    explicit_poll = sum(1 for t in texts if _POLL_EXPLICIT.search(t))
    poll_format = explicit_poll / n  # doit être ≥ 0.3 pour activer

    # Guard dur : si pas assez de vrais formats de poll → score 0
    if poll_format < 0.15:
        return 0.0

    # 2. Diversité des hashtags (min 20 total → il faut du volume réel)
    all_tags = [h for t in texts for h in re.findall(r'#(\w+)', t.lower())]
    if len(all_tags) < 15:
        return 0.0  # Pas assez de hashtags pour être un poll account légitime

    diversity = len(set(all_tags)) / len(all_tags)
    if diversity < 0.20:
        return 0.0  # Low diversity = bot thématique, pas poll humain

    # 3. Aucun CTA → distingue des spammers
    cta_posts = sum(1 for t in texts if _CTA_RE.search(t))
    no_cta = 1.0 if cta_posts <= 1 else max(0, 1 - cta_posts / n)

    # 4. Bio personnelle décontractée
    bio_personal = 0.0
    if bio:
        has_emoji = bool(re.search(r'[\U00010000-\U0010FFFF\U0001F300-\U0001FFFF]', bio))
        is_casual  = len(bio) < 160 and not _CTA_RE.search(bio)
        bio_personal = 0.5 * has_emoji + 0.5 * is_casual

    score = (
        poll_format    * 0.40 +
        diversity      * 0.35 +
        no_cta         * 0.15 +
        bio_personal   * 0.10
    )
    return float(np.clip(score, 0, 1))


# ─── Archétype 2 : Quote / RP / Parody Human ─────────────────────────────────

def _rp_human_v2(bio: str, texts: list) -> float:
    """
    Signatures exactes de @rise_quotes :
      - bio contient une déclaration RP spécifique ('quotes taken from', 'fan acc')
        MAIS PAS juste 'not affiliated' ou 'bot' seul (trop générique).
      - zero URL dans les posts (un vrai bot spam a des liens)
      - zero CTA promotionnel
      - cohérence thématique forte (même vocabulaire/univers fictif)
    Guard : le bot @CappedPair a bio 'NOT AFFILIATED WITH ANYONE' mais des posts
    hétérogènes sans cohérence de citations → doit scorer 0.
    """
    if not bio and not texts:
        return 0.0

    # 1. Déclaration RP dans la bio (signal fort et STRICT)
    rp_bio = 1.0 if _RP_BIO_RE.search(bio) else 0.0

    if rp_bio == 0:
        return 0.0  # Sans déclaration RP explicite, ne pas activer ce protecteur

    # 2. Aucune URL = RP pur (pas de spam)
    url_posts = sum(1 for t in texts if _URL_RE.search(t))
    url_ratio = url_posts / max(len(texts), 1)
    no_url_score = 1.0 if url_ratio < 0.1 else max(0, 1 - url_ratio * 2)

    # Guard URL strict : si > 20% des posts ont des URLs → pas un vrai RP
    if url_ratio > 0.20:
        return 0.0

    # 3. Aucun CTA
    cta_posts = sum(1 for t in texts if _CTA_RE.search(t))
    no_cta = 1.0 if cta_posts == 0 else 0.5

    # Guard CTA : si plus de 2 posts CTA → pas un vrai RP
    if cta_posts > 2:
        return 0.0

    # 4. Cohérence thématique FORTE : vocabulaire répété entre posts (même univers)
    all_words = []
    for t in texts:
        clean = re.sub(r'https?\S+|#\w+|@\w+', '', t).lower()
        all_words.extend(re.findall(r'\b[a-z]{3,}\b', clean))

    thematic_coherence = 0.0
    if len(all_words) >= 10:
        counts = Counter(all_words)
        top10  = sum(v for _, v in counts.most_common(10))
        thematic_coherence = min(top10 / len(all_words), 1.0)

    # Guard thématique : un vrai RP a une haute cohérence (> 0.25)
    # Un bot généraliste avec des posts hétérogènes aura une faible cohérence
    if thematic_coherence < 0.15:
        return 0.0

    score = (
        rp_bio               * 0.45 +
        no_url_score         * 0.25 +
        thematic_coherence   * 0.20 +
        no_cta               * 0.10
    )
    return float(np.clip(score, 0, 1))


# ─── Archétype 3 : Stan / Fandom Human ───────────────────────────────────────

def _stan_human_v2(bio: str, texts: list) -> float:
    """
    Signatures exactes de @sapalocha98 :
      - Mix de blagues (puns) + promotion fandom dans les mêmes posts
        → le bot pur n'alterne pas entre humour et promo
      - Très peu de diversité de hashtags (ultra-fandom = 2-3 tags répétés)
        → inverse du poll : le stan a un seul sujet
      - Bio avec vocabulaire stan / fandom explicite
      - Premier personne (first-person dans les blagues)
    """
    if not texts:
        return 0.0

    n = len(texts)

    # 1. Présence de jeux de mots / puns (signal humain fort et rare chez les bots)
    pun_posts  = sum(1 for t in texts if _PUN_SETUP.search(t))
    pun_ratio  = pun_posts / n

    # 2. Vocabulaire fandom dans bio + posts
    bio_words  = set(re.findall(r'\b\w+\b', bio.lower()))
    post_words = set()
    for t in texts:
        post_words.update(re.findall(r'\b\w+\b', t.lower()))
    fandom_hit = len((bio_words | post_words) & _STAN_MARKERS)
    fandom_score = min(fandom_hit / 3.0, 1.0)  # 3 mots fandom = 1.0

    # 3. Diversité de hashtags très faible = ultra-fandom (≠ spam générique)
    all_tags = [h for t in texts for h in re.findall(r'#(\w+)', t.lower())]
    if len(all_tags) >= 5:
        tag_diversity = len(set(all_tags)) / len(all_tags)
        # Faible diversité (< 0.15) = bon signal stan (ultra-focused)
        fandom_focus = 1.0 - min(tag_diversity / 0.15, 1.0)
    else:
        fandom_focus = 0.0

    # 4. Mix puns + promo dans le même post = signature humaine distinctive
    mixed_posts = 0
    for t in texts:
        has_pun  = bool(_PUN_SETUP.search(t))
        has_promo = bool(re.search(r'#\w+', t) and fandom_hit > 0)
        if has_pun and has_promo:
            mixed_posts += 1
    mix_ratio = mixed_posts / max(n, 1)

    # Stan nécessite au moins du fandom ET des blagues pour être protégé
    if fandom_score < 0.1 and pun_ratio < 0.05:
        return 0.0

    score = (
        pun_ratio     * 0.35 +
        fandom_score  * 0.25 +
        fandom_focus  * 0.20 +
        mix_ratio     * 0.20
    )
    return float(np.clip(score, 0, 1))


# ─── Extracteur Principal ──────────────────────────────────────────────────────

def extract_lrh_residual(
    u_df: pd.DataFrame,
    p_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retourne un DataFrame {user_id, lrh2_*} avec les scores résiduels.
    Feature protectrice uniquement. Aucune logique d'accusation.
    """
    if u_df.empty:
        return pd.DataFrame()

    uid_col = "user_id"

    # Grouper posts par uid
    posts_by_uid: dict = {str(uid): [] for uid in u_df[uid_col].values}
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

        sc_poll  = _poll_human_v2(texts, bio)
        sc_rp    = _rp_human_v2(bio, texts)
        sc_stan  = _stan_human_v2(bio, texts)

        # Score résiduel = max des 3 protecteurs (un seul doit s'activer)
        residual = max(sc_poll, sc_rp, sc_stan)

        records.append({
            uid_col:              uid,
            "lrh2_poll_score":    round(sc_poll,   3),
            "lrh2_rp_score":      round(sc_rp,     3),
            "lrh2_stan_score":    round(sc_stan,    3),
            "lrh2_residual_score": round(residual,  3),
        })

    return pd.DataFrame(records)
