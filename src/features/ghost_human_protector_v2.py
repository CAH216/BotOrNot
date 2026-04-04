# -*- coding: utf-8 -*-
"""
src/features/ghost_human_protector_v2.py

Ghost Human Protector V2
=========================
Produit un score de "bénignité humaine" multi-dimensionnel.
Cible : live-tweeters, parieurs, comptes fantômes organiques, faibles activités réelles.

Principe : on ne décide pas ici — on produit des features numériques brutes que
LightGBM/CatBoost apprendront à pondérer. Aucun seuil décisionnel.

Familles couvertes
──────────────────
A. Variété inter-posts (format, longueur, ponctuation)
B. Bruit humain local (typos, mixed-case, ellipses, abréviations)
C. Alternance de ton / polarité / registre
D. Non-template behavior (entropy de structure)
E. Mismatch avec les patrons bots connus (regularité mécanique)
F. Protections spéciales ghost-sleeper (0-1 posts, pas de pénalité)
"""
import re
import math
import unicodedata
import numpy as np
import pandas as pd


# ─── Helpers textuels ──────────────────────────────────────────────────────────

_URL_RE   = re.compile(r'https?://\S+|www\.\S+')
_MENTION  = re.compile(r'@\w+')
_HASHTAG  = re.compile(r'#\w+')
_EMOJI_RE = re.compile(
    "[\U00010000-\U0010FFFF"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF\u2700-\u27BF]+",
    flags=re.UNICODE,
)

def _strip_noise(text: str) -> str:
    """Retire URLs, mentions, hashtags pour analyser le corps réel."""
    t = _URL_RE.sub(" ", text)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(" ", t)
    return t.strip()

def _char_entropy(text: str) -> float:
    """Shannon entropy sur les caractères."""
    if not text:
        return 0.0
    counts = {}
    for c in text:
        counts[c] = counts.get(c, 0) + 1
    n = len(text)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())

def _punct_diversity(text: str) -> float:
    """Proportion de types de ponctuation distincts parmi {.,!?;:-}."""
    marks = set(c for c in text if c in ".,!?;:-")
    return len(marks) / 6.0

def _word_count(text: str) -> int:
    return len(text.split())

def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def _has_mixed_case(text: str) -> int:
    """Détecte camelCase / mid-sentence caps (humain qui s'emballe)."""
    return int(bool(re.search(r'[a-z][A-Z]', text)))

def _has_typo_pattern(text: str) -> int:
    """Répétitions de lettres (aaah, looool, yeahhh) — signal humain."""
    return int(bool(re.search(r'(.)\1{2,}', text)))

def _has_self_correction(text: str) -> int:
    """*corr, je veux dire, typo: — micro-signaux de correction humaine."""
    patterns = [r'\*\w+', r'\bcorr\b', r'\btypo\b', r'je veux dire', r'i mean']
    return int(any(re.search(p, text, re.IGNORECASE) for p in patterns))

def _has_informal_abbrev(text: str) -> int:
    """lol, omg, wtf, lmao, tbh, imo, ngl, smh — argot organique."""
    abbrevs = {'lol', 'omg', 'wtf', 'lmao', 'lmfao', 'tbh', 'imo', 'ngl',
                'smh', 'irl', 'gg', 'oof', 'brb', 'afk', 'rofl', 'xd', 'mdr',
                'ptdr', 'jsp', 'jtm', 'bg', 'ouf'}
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return int(bool(words & abbrevs))

def _has_question(text: str) -> int:
    return int('?' in text)

def _has_exclamation(text: str) -> int:
    return int('!' in text)

def _emoji_count(text: str) -> int:
    return len(_EMOJI_RE.findall(text))

def _structure_skeleton(text: str) -> str:
    """
    Squelette structurel minimal pour détecter les templates.
    Remplace mots par W, chiffres par N, symboles par S.
    """
    t = _URL_RE.sub("URL", text)
    t = _MENTION.sub("@", t)
    t = _HASHTAG.sub("#", t)
    t = re.sub(r'\d+', 'N', t)
    t = re.sub(r'[a-zA-Z\u00C0-\u024F]+', 'W', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


# ─── Extracteur par post ────────────────────────────────────────────────────────

def _post_features(text: str) -> dict:
    body = _strip_noise(text)
    return {
        "length": len(text),
        "word_count": _word_count(body),
        "char_entropy": _char_entropy(body),
        "punct_diversity": _punct_diversity(text),
        "upper_ratio": _upper_ratio(body),
        "has_mixed_case": _has_mixed_case(text),
        "has_typo_pattern": _has_typo_pattern(text),
        "has_self_correction": _has_self_correction(text),
        "has_informal_abbrev": _has_informal_abbrev(text),
        "has_question": _has_question(text),
        "has_exclamation": _has_exclamation(text),
        "emoji_count": _emoji_count(text),
        "skeleton": _structure_skeleton(text),
    }


# ─── Agrégation utilisateur ─────────────────────────────────────────────────────

def _user_level_features(post_rows: list[dict]) -> dict:
    """
    Prend la liste des dicts de features par post pour un utilisateur.
    Retourne un dict de features agrégées.
    """
    n = len(post_rows)

    if n == 0:
        # Ghost sleeper : aucun post → on ne le pénalise pas
        return {
            "gh_n_posts": 0,
            "gh_len_std": 0.0,          # pas de donnée → neutre
            "gh_len_cv": 0.0,
            "gh_entropy_mean": 0.0,
            "gh_entropy_std": 0.0,
            "gh_punct_div_mean": 0.0,
            "gh_upper_ratio_std": 0.0,
            "gh_frac_mixed_case": 0.0,
            "gh_frac_typo_pattern": 0.0,
            "gh_frac_self_correction": 0.0,
            "gh_frac_informal_abbrev": 0.0,
            "gh_frac_question": 0.0,
            "gh_frac_exclamation": 0.0,
            "gh_emoji_variety": 0.0,
            "gh_skeleton_unique_ratio": 1.0,  # 1 = pas de clone → humain par défaut
            "gh_tone_alternation": 0.0,
            "gh_non_template_score": 1.0,     # bénin par défaut
            "gh_organic_score": 1.0,
        }

    lengths    = [r["length"]        for r in post_rows]
    entropies  = [r["char_entropy"]  for r in post_rows]
    punct_divs = [r["punct_diversity"] for r in post_rows]
    upper_rats = [r["upper_ratio"]   for r in post_rows]

    len_mean = float(np.mean(lengths)) if lengths else 1.0
    len_std  = float(np.std(lengths))
    len_cv   = len_std / (len_mean + 1e-9)  # coefficient de variation

    entr_mean = float(np.mean(entropies))
    entr_std  = float(np.std(entropies))

    # ── A. Variété inter-posts ────────────────────────────────────────────────
    # Coefficient de variation de longueur : bot = 0 (tous identiques)
    gh_len_cv = min(len_cv, 5.0) / 5.0  # normalisé 0-1

    # ── B. Bruit humain ───────────────────────────────────────────────────────
    gh_frac_mixed     = float(np.mean([r["has_mixed_case"]      for r in post_rows]))
    gh_frac_typo      = float(np.mean([r["has_typo_pattern"]    for r in post_rows]))
    gh_frac_selfcorr  = float(np.mean([r["has_self_correction"] for r in post_rows]))
    gh_frac_informal  = float(np.mean([r["has_informal_abbrev"] for r in post_rows]))

    # ── C. Alternance de ton ──────────────────────────────────────────────────
    # Passage question ↔ exclamation ↔ neutre entre posts consécutifs
    tones = []
    for r in post_rows:
        if r["has_question"] and not r["has_exclamation"]:
            tones.append(0)
        elif r["has_exclamation"] and not r["has_question"]:
            tones.append(2)
        else:
            tones.append(1)
    tone_changes = sum(1 for i in range(1, len(tones)) if tones[i] != tones[i-1])
    gh_tone_alternation = tone_changes / max(n - 1, 1)

    # ── D. Non-template behavior ──────────────────────────────────────────────
    skeletons  = [r["skeleton"] for r in post_rows]
    unique_sk  = len(set(skeletons))
    sk_unique_ratio = unique_sk / n   # 1.0 = tous uniques = humain

    # ── E. Mismatch avec patron bot (mécanique régulière) ────────────────────
    # Un bot a: entropy std ≈ 0 (textes identiques), len_cv ≈ 0, no informal abbrev
    # On cherche le contraire
    gh_non_template = (entr_std + gh_len_cv + gh_frac_informal) / 3.0

    # ── Emoji variety ─────────────────────────────────────────────────────────
    all_emoji_counts = [r["emoji_count"] for r in post_rows]
    # Diversité : avoir parfois 0 et parfois 3 = humain éclectique
    gh_emoji_variety = float(np.std(all_emoji_counts)) / (np.mean(all_emoji_counts) + 1e-9)
    gh_emoji_variety = min(gh_emoji_variety, 5.0) / 5.0

    # ── Score composite de bénignité ──────────────────────────────────────────
    # C'est une FEATURE pour LightGBM, pas un décideur
    gh_organic = (
        gh_len_cv               * 0.20 +
        entr_std                * 0.15 +
        gh_frac_typo            * 0.10 +
        gh_frac_informal        * 0.10 +
        gh_tone_alternation     * 0.15 +
        sk_unique_ratio         * 0.20 +
        gh_frac_mixed           * 0.10
    )
    gh_organic = float(np.clip(gh_organic, 0.0, 1.0))

    return {
        "gh_n_posts":               n,
        "gh_len_std":               len_std,
        "gh_len_cv":                gh_len_cv,
        "gh_entropy_mean":          entr_mean,
        "gh_entropy_std":           entr_std,
        "gh_punct_div_mean":        float(np.mean(punct_divs)),
        "gh_upper_ratio_std":       float(np.std(upper_rats)),
        "gh_frac_mixed_case":       gh_frac_mixed,
        "gh_frac_typo_pattern":     gh_frac_typo,
        "gh_frac_self_correction":  gh_frac_selfcorr,
        "gh_frac_informal_abbrev":  gh_frac_informal,
        "gh_frac_question":         float(np.mean([r["has_question"]    for r in post_rows])),
        "gh_frac_exclamation":      float(np.mean([r["has_exclamation"] for r in post_rows])),
        "gh_emoji_variety":         gh_emoji_variety,
        "gh_skeleton_unique_ratio": sk_unique_ratio,
        "gh_tone_alternation":      gh_tone_alternation,
        "gh_non_template_score":    gh_non_template,
        "gh_organic_score":         gh_organic,
    }


# ─── Point d'entrée public ─────────────────────────────────────────────────────

def extract_ghost_human_protector_v2(
    u_df: pd.DataFrame,
    p_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retourne un DataFrame {user_id, gh_*} avec les features de bénignité humaine.
    Toutes les valeurs sont numériques et normalisées [0, 1] ou en unités naturelles.
    Les ghost sleepers (0 post) reçoivent des valeurs neutres / bénignes par défaut.
    """
    if u_df.empty:
        return pd.DataFrame()

    uid_col = "user_id"

    # Construit index user_id → liste de dicts post
    user_posts: dict[str, list[dict]] = {
        str(uid): [] for uid in u_df[uid_col].values
    }

    if not p_df.empty and "text" in p_df.columns:
        p_work = p_df[[uid_col, "text"]].copy()
        p_work["text"] = p_work["text"].fillna("").astype(str)
        for _, row in p_work.iterrows():
            uid = str(row[uid_col])
            if uid in user_posts:
                user_posts[uid].append(_post_features(row["text"]))

    # Agrège par utilisateur
    records = []
    for uid in u_df[uid_col].values:
        feat = _user_level_features(user_posts[str(uid)])
        feat[uid_col] = uid
        records.append(feat)

    res = pd.DataFrame(records)

    # Nettoyage
    feat_cols = [c for c in res.columns if c != uid_col]
    for c in feat_cols:
        res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0.0)

    # Réordonne pour que user_id soit en premier
    cols = [uid_col] + feat_cols
    return res[cols]
