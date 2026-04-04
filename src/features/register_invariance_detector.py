#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
register_invariance_detector.py — Stealth Bot Fingerprint

THÉORIE : Les bots sophistiqués s'imitent les humains en surface
(bio remplie, posts variés en longueur, horaires normaux) mais
trahissent leur nature par une INVARIANCE profonde du registre
linguistique. Un humain normal DÉRIVE — son style change selon
l'heure, l'humeur, le contexte. Un bot reste trop stable.

Ce module détecte cette sur-stabilité pathologique via 6 signaux :

1. rid_register_cv
   Coefficient de variation du ratio majuscules par post.
   Humains : très variable (début de phrase, enthousiasme, colère).
   Bots : stable (template rendu).

2. rid_punct_stability
   Écart-type normalisé des signes de ponctuation par post.
   Humains : dérivent (parfois virgules, parfois tirets, parfois rien).
   Bots : ponctuation identique d'un post à l'autre.

3. rid_ttr_variance
   Variance du Type-Token Ratio (richesse lexicale) par post.
   Humains : TTR fluctue naturellement.
   Bots paraphraseurs : TTR identique (même vocabulaire recyclé).

4. rid_interval_regularity
   1 / (CV des intervalles inter-posts + ε).
   Bots cron-job : intervalles réguliers → CV très bas → score haut.
   Humains : tweetent quand ils veulent → CV élevé → score bas.

5. rid_skeleton_entropy
   Entropie de Shannon des squelettes structuraux des posts.
   Bots : entropy faible (même structure répétée).
   Humains : entropy haute (structures diverses).
   NB : complémentaire au ghost slim (ratio unique, ici entropy globale).

6. rid_topic_lock
   Cosine similarity inter-posts (bag-of-words normalisé).
   Bots : tous les posts parlent du même sujet → similarité haute.
   Humains : sujets variables → similarité basse.

Composite : rid_stealth_score = weighted sum des signaux anormaux.

ACTIVATION : Désactivé par défaut dans le monolithe via flag
  config["use_stealth_bot_fingerprint"] = False

Ne pas activer sans validation benchmark complète 5-gates.
"""

import re
import math
import numpy as np
import pandas as pd
from collections import Counter


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _ttr(text: str) -> float:
    """Type-Token Ratio : tokens uniques / total tokens."""
    tokens = text.lower().split()
    if len(tokens) < 3:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _punct_count(text: str) -> int:
    return len(re.findall(r'[,.!?;:\-–]', text))


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _skeleton(text: str) -> str:
    """Squelette structurel simplifié (compatible Ghost Slim)."""
    t = re.sub(r'https?\S+', 'URL', text)
    t = re.sub(r'@\w+', 'USER', t)
    t = re.sub(r'#\w+', 'TAG', t)
    t = re.sub(r'\d+', 'NUM', t)
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return ' '.join(word[0] for word in t.split()[:8])


def _bow_similarity(texts: list[str]) -> float:
    """Cosine similarity moyenne entre toutes les paires de posts (BoW)."""
    if len(texts) < 2:
        return 0.0
    # Construit les vecteurs BoW
    vocab    = set()
    counters = []
    for t in texts:
        c = Counter(t.lower().split())
        counters.append(c)
        vocab.update(c.keys())
    vocab = list(vocab)
    if not vocab:
        return 0.0

    def make_vec(c):
        v = np.array([c.get(w, 0) for w in vocab], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    vecs = [make_vec(c) for c in counters]
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(float(np.dot(vecs[i], vecs[j])))
    return float(np.mean(sims)) if sims else 0.0


def _shannon_entropy(items: list) -> float:
    if not items:
        return 0.0
    c = Counter(items)
    n = len(items)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in c.values() if cnt > 0)


# ── Extracteur principal ──────────────────────────────────────────────────────

def extract_register_invariance(u_df: pd.DataFrame,
                                p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame avec les colonnes rid_* par user_id.
    Utilisé comme bloc de features injecté dans le monolithe (désactivé par défaut).
    """
    # Prépare les posts
    if p_df.empty or "text" not in p_df.columns:
        res = pd.DataFrame({"user_id": u_df["user_id"]})
        for col in ["rid_register_cv", "rid_punct_stability", "rid_ttr_variance",
                    "rid_interval_regularity", "rid_skeleton_entropy",
                    "rid_topic_lock", "rid_stealth_score"]:
            res[col] = 0.0
        return res

    p = p_df.copy()
    if "author_id" in p.columns and "user_id" not in p.columns:
        p = p.rename(columns={"author_id": "user_id"})
    p["text"] = p["text"].fillna("").astype(str)

    # Timestamps pour l'intervalle régularité
    has_time = "created_at" in p.columns
    if has_time:
        p["_ts"] = pd.to_datetime(p["created_at"], errors="coerce")

    records = []
    for uid, grp in p.groupby("user_id"):
        texts = grp["text"].tolist()
        n     = len(texts)

        # ── 1. Register CV (variation du ratio majuscules) ────────────────────
        uppers = [_upper_ratio(t) for t in texts]
        mean_u = np.mean(uppers) if uppers else 0
        std_u  = np.std(uppers)  if uppers else 0
        register_cv = (std_u / (mean_u + 1e-6)) if n >= 2 else 0.0
        # Inversion : cv FAIBLE = suspect (bot stable)
        # On garde la valeur brute, le modèle apprend le signe

        # ── 2. Ponctuation stability (std normalisée) ─────────────────────────
        puncts = [_punct_count(t) / (len(t.split()) + 1) for t in texts]
        punct_stability = float(np.std(puncts)) if n >= 2 else 0.0

        # ── 3. TTR Variance ───────────────────────────────────────────────────
        ttrs = [_ttr(t) for t in texts if len(t.split()) >= 3]
        ttr_variance = float(np.var(ttrs)) if len(ttrs) >= 2 else 0.0

        # ── 4. Intervalle régularité ──────────────────────────────────────────
        if has_time and n >= 3:
            ts = grp["_ts"].dropna().sort_values()
            if len(ts) >= 3:
                diffs = pd.Series(ts.diff().dt.total_seconds().dropna().tolist())
                diffs = diffs[diffs > 0]
                if len(diffs) >= 2:
                    m_d = diffs.mean()
                    s_d = diffs.std()
                    cv  = s_d / (m_d + 1e-6)
                    interval_regularity = float(1.0 / (cv + 1e-3))
                else:
                    interval_regularity = 0.0
            else:
                interval_regularity = 0.0
        else:
            interval_regularity = 0.0

        # ── 5. Skeleton Entropy ───────────────────────────────────────────────
        skeletons     = [_skeleton(t) for t in texts]
        skel_entropy  = _shannon_entropy(skeletons)
        # Entropie basse = peu de structures → bot

        # ── 6. Topic Lock (BoW cosine) ────────────────────────────────────────
        topic_lock = _bow_similarity(texts) if n >= 2 else 0.0

        # ── Composite score ───────────────────────────────────────────────────
        # Chaque composante "anormale" (bot-like) contribue positivement
        # Le modèle apprend les poids — on ne force pas le signe ici
        stealth_score = (
            (1 / (register_cv + 1e-3)) * 0.20 +    # faible variance = suspect
            (1 / (punct_stability + 1e-3)) * 0.15 + # faible variance = suspect
            (1 / (ttr_variance + 1e-3)) * 0.15 +    # faible variance = suspect
            interval_regularity * 0.20 +             # régularité = suspect
            (1 / (skel_entropy + 1e-3)) * 0.15 +    # faible entropy = suspect
            topic_lock * 0.15                        # topic-locked = suspect
        )

        records.append({
            "user_id":                str(uid),
            "rid_register_cv":        round(register_cv, 6),
            "rid_punct_stability":    round(punct_stability, 6),
            "rid_ttr_variance":       round(ttr_variance, 6),
            "rid_interval_regularity": round(interval_regularity, 6),
            "rid_skeleton_entropy":   round(skel_entropy, 6),
            "rid_topic_lock":         round(topic_lock, 6),
            "rid_stealth_score":      round(stealth_score, 6),
        })

    rid_df = pd.DataFrame(records)
    if rid_df.empty:
        rid_cols = ["rid_register_cv", "rid_punct_stability", "rid_ttr_variance",
                    "rid_interval_regularity", "rid_skeleton_entropy",
                    "rid_topic_lock", "rid_stealth_score"]
        rid_df = pd.DataFrame({"user_id": u_df["user_id"].astype(str)})
        for c in rid_cols:
            rid_df[c] = 0.0
        return rid_df

    # Fusionne avec tous les users pour éviter les manquants
    all_users = pd.DataFrame({"user_id": u_df["user_id"].astype(str)})
    merged    = all_users.merge(rid_df, on="user_id", how="left").fillna(0)
    return merged
