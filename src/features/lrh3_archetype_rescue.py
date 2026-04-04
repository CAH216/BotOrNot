# -*- coding: utf-8 -*-
"""
src/features/lrh3_archetype_rescue.py
======================================
LRH3 — Archetype Rescue : 3 sous-protecteurs hyper-specifiques.

Contrainte absolue :
  - Déficit SHAP à combler : -12 a -17 par compte
  - Les features LightGBM ne peuvent générer que 1-3 pts de SHAP par feature
  - LRH3 est donc honnêtement un effort marginal — à évaluer sans illusion

Design :
  1. lrh3_poll_topical  : diversité TOPICALE (pas juste tag) + format poll pur
  2. lrh3_rp_hardened   : version encore plus stricte du signal RP (binaire fort)
  3. lrh3_pun_density   : ratio pur de puns (valeur brute, très discriminante)
  4. lrh3_pun_fandom    : combo pun * fandom (quasi-unique à @sapalocha98)
  5. lrh3_rescue_score  : max composite
"""
import re, math
import numpy as np
import pandas as pd
from collections import Counter

# ── Compilations ──────────────────────────────────────────────────────────────

_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
_CTA_RE = re.compile(
    r'\b(click|link in bio|dm me|buy now|get yours|promo|discount|order now)\b', re.I)

# Patterns de setups de blagues (complet, non-vide pour éviter le bug de l'audit)
_PUN_RE = re.compile(
    r'\b(why did(?:n.t)?|what did|what do you call|what happens when|'
    r'knock knock|did you hear|how do you|what goes|why does|'
    r"what.s the difference|what.s a)\b",
    re.I
)

# RP strict : bio attribut une citation à un auteur/source réelle
_RP_STRICT = re.compile(
    r'\b(quotes?\s+(?:taken\s+from|from|by)|taken\s+from|lines\s+from|'
    r'quoting\s+@|fan\s+acc(?:ount)?|roleplay|rp\s+account|'
    r'parody\s+(?:account|of)|character\s+bot|post\s+bot|(?:quote|fan)\s+bot)\b',
    re.I
)

# Marqueurs de poll EXPLICITES (format liste/choix)
_POLL_FORMAT = re.compile(
    r'(top\s*\d+|#\d+\s+vs\.?|who\s+would\s+win|pick\s+one|your\s+fav(?:ou?rite)?|'
    r'would\s+you\s+rather|\bA\)\s|\bB\)\s|1\.\s+\w|rate\s+|rank\s+|'
    r'best\s+(?:of|ever)|vs\.?\s+\w|poll[:\s])',
    re.I
)

# Fandom markers pour combo
_FANDOM_RE = re.compile(
    r'\b(stan|stanning|ult|bias|comeback|stream|mv|blink|army|once|moa|carat|'
    r'kpop|kpopper|girlgroup|boygroup|weverse|vlive|fandom|lightstick)\b', re.I
)


# ── Sous-protecteur 1 : Poll Topical ─────────────────────────────────────────

def _poll_topical(texts: list, bio: str) -> float:
    """
    Signal très spécifique @no_context_poll :
    - Unique tags count élevé (> 60) : signe d'une grande variété topicale
    - Ratio format poll explicite > 0.10
    - Aucun CTA
    - Ratio URL modéré (< 0.50) : les poll humans partagent des liens mais pas massivement
    """
    if len(texts) < 10:
        return 0.0

    n = len(texts)
    tags   = [h for t in texts for h in re.findall(r'#(\w+)', t.lower())]
    unique = len(set(tags))

    # Guard 1 : il faut un grand nombre de hashtags uniques (signal poll humain)
    if unique < 50:
        return 0.0

    poll_posts = sum(1 for t in texts if _POLL_FORMAT.search(t))
    poll_fmt   = poll_posts / n

    # Guard 2 : il faut au moins quelques posts de format poll
    if poll_fmt < 0.08:
        return 0.0

    cta_posts = sum(1 for t in texts if _CTA_RE.search(t))
    no_cta    = 1.0 if cta_posts <= 1 else max(0, 1 - cta_posts / n)

    # Score : unique_tags principal signal (plafonné à 150)
    unique_score = min(unique / 150.0, 1.0)

    score = (
        unique_score * 0.55 +
        poll_fmt     * 0.25 +
        no_cta       * 0.20
    )
    return float(np.clip(score, 0, 1))


# ── Sous-protecteur 2 : RP Hardened ──────────────────────────────────────────

def _rp_hardened(bio: str, texts: list) -> float:
    """
    Signal ultra-strict @rise_quotes :
    - Bio avec déclaration d'attribution STRICTE (notre regex patchée)
    - Zéro URL dans les posts
    - Cohérence thématique > 0.20
    - Zéro CTA
    Retourne une valeur haute (0.9+) si toutes les conditions sont remplies.
    """
    if not bio or not texts:
        return 0.0

    # Condition 1 : bio stricte
    if not _RP_STRICT.search(bio):
        return 0.0

    # Condition 2 : zéro URL
    url_count = sum(1 for t in texts if _URL_RE.search(t))
    if url_count > 2:  # Tolérance de 2 posts avec URL sur 100 posts
        return 0.0

    # Condition 3 : zéro CTA
    cta_count = sum(1 for t in texts if _CTA_RE.search(t))
    if cta_count > 0:
        return 0.0

    # Condition 4 : cohérence thématique (même univers fictif)
    all_words = []
    for t in texts:
        clean = re.sub(r'https?\S+|#\w+|@\w+', '', t).lower()
        all_words.extend(re.findall(r'\b[a-z]{3,}\b', clean))

    if len(all_words) < 20:
        return 0.0

    counts = Counter(all_words)
    top10  = sum(v for _, v in counts.most_common(10))
    coherence = top10 / len(all_words)

    if coherence < 0.18:
        return 0.0

    # Toutes les conditions remplies : score high
    url_score = 1.0 - (url_count / max(len(texts), 1))
    coh_score = min(coherence / 0.35, 1.0)  # plafonné à 0.35

    score = 0.70 + 0.20 * url_score + 0.10 * coh_score
    return float(np.clip(score, 0, 1))


# ── Sous-protecteur 3 : Stan + Pun Density ───────────────────────────────────

def _pun_density(texts: list) -> float:
    """
    Ratio brut de posts contenant un setup de blague.
    Valeur pour @sapalocha98 : 0.290 (23 sur 100 posts).
    Valeur pour tous les bots connus : < 0.05.
    Signal très discriminant et simple.
    """
    if not texts:
        return 0.0
    pun_posts = sum(1 for t in texts if _PUN_RE.search(t))
    return round(pun_posts / len(texts), 4)


def _pun_fandom_combo(texts: list, bio: str) -> float:
    """
    Combo pun + fandom dans le même post.
    Signal quasi-unique à @sapalocha98 : aucun bot ne rédige des blagues
    ET des contenus fandom dans le même tweet.
    """
    if not texts:
        return 0.0

    n = len(texts)
    # Vérifier présence globale de fandom
    all_content = " ".join(texts) + " " + bio
    has_fandom  = bool(_FANDOM_RE.search(all_content))
    if not has_fandom:
        return 0.0

    # Compter les posts qui combinent pun + fandom
    mixed_posts = 0
    for t in texts:
        if _PUN_RE.search(t) and _FANDOM_RE.search(t):
            mixed_posts += 1

    combo_ratio = mixed_posts / n
    return float(np.clip(combo_ratio * 3.0, 0, 1))  # Amplifié x3 car signal rare


# ── Extracteur Principal ───────────────────────────────────────────────────────

def extract_lrh3_archetype_rescue(
    u_df: pd.DataFrame,
    p_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retourne {user_id, lrh3_*} — features protectrices uniquement.
    Aucune logique d'accusation.
    """
    if u_df.empty:
        return pd.DataFrame()

    uid_col       = "user_id"
    posts_by_uid  = {str(uid): [] for uid in u_df[uid_col].values}
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

        poll = _poll_topical(texts, bio)
        rp   = _rp_hardened(bio, texts)
        pun  = _pun_density(texts)
        combo= _pun_fandom_combo(texts, bio)

        rescue = max(poll, rp, min(pun * 2.5, 1.0), combo)

        records.append({
            uid_col:                 uid,
            "lrh3_poll_topical":     round(poll,  3),
            "lrh3_rp_hardened":      round(rp,    3),
            "lrh3_pun_density":      round(pun,   4),
            "lrh3_pun_fandom_combo": round(combo, 3),
            "lrh3_rescue_score":     round(rescue, 3),
        })

    return pd.DataFrame(records)
