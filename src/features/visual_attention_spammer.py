#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visual_attention_spammer.py — Détecteur de Bots d'Affiliation / Clickbait (VAS)

THÉORIE : Les bots de spam, crypto, paris sportifs ou d'amplification
sont conçus pour capter l'attention dans un flux (scroll). Ils abusent
des marqueurs visuels "d'action" ou "d'alerte", là où les humains 
utilisent l'expressivité pour communiquer une émotion.

Signaux ciblés :
1. Emojis d'Attention (🚨, 📌, 👉, 💥, 🚀, 💰) vs Emojis Émotionnels (😂, 😭, 💀, 👀).
2. Typographie d'urgence (mots en ALL_CAPS : FREE, WIN, CLICK).
3. Call-to-actions ("link in bio", "thread", "read more").

Si on trouve des humains très agités (live-tweeters) avec beaucoup
de majuscules, ils seront sauvés s'ils utilisent des emojis d'émotion
plutôt que d'attention commerciale.
"""

import re
import pandas as pd
import numpy as np

# ── Dictionnaires de Signaux ──────────────────────────────────────────────────

ATTENTION_EMOJIS = {
    '🚨', '📌', '👉', '👇', '💥', '🔥', '🏆', '💰', '🚀', '📢', '🔊', '⚠️', '✅', 
    '➡️', '▶️', '❗', '❓', '💎', '🎁', '💸', '📈', '📍', '🔴', '🟢'
}

EMOTION_EMOJIS = {
    '😂', '😭', '💀', '👀', '🥺', '❤️', '🙏', '😂', '🤣', '😅', '🙃', '😍', '🥰',
    '😘', '🤔', '🙄', '😏', '😔', '😡', '🤬', '💔', '✨', '🙌', '👏', '🤦', '🤷'
}

CLICKBAIT_REGEX = re.compile(
    r'\b(link in bio|click here|read more|thread|dm me|giveaway|sign up|subscribe)\b', 
    re.IGNORECASE
)

# ── Extracteur ────────────────────────────────────────────────────────────────

def extract_visual_attention_spammer(u_df: pd.DataFrame, p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère le bloc de features VAS (Visual Attention Spammer) pour chaque user.
    """
    if p_df.empty or 'text' not in p_df.columns:
        res = pd.DataFrame({"user_id": u_df["user_id"]})
        cols = ["vas_attention_ratio", "vas_emotion_ratio", "vas_emoji_spam_idx",
                "vas_all_caps_ratio", "vas_clickbait_ratio", "vas_spam_score"]
        for c in cols:
            res[c] = 0.0
        return res

    p = p_df.copy()
    if "author_id" in p.columns and "user_id" not in p.columns:
        p = p.rename(columns={"author_id": "user_id"})
    
    p["text"] = p["text"].fillna("").astype(str)
    
    records = []
    
    for uid, grp in p.groupby("user_id"):
        texts = grp["text"].tolist()
        n     = len(texts)
        if n == 0:
            continue
            
        attn_count = 0
        emot_count = 0
        all_caps_words = 0
        total_words = 0
        clickbait_hits = 0
        
        for t in texts:
            # Emoji parsing
            chars = list(t)
            for c in chars:
                if c in ATTENTION_EMOJIS:
                    attn_count += 1
                elif c in EMOTION_EMOJIS:
                    emot_count += 1
                    
            # All caps words
            words = re.findall(r'\b[a-zA-Z]+\b', t)
            total_words += len(words)
            for w in words:
                if len(w) > 3 and w.isupper():
                    all_caps_words += 1
                    
            # Clickbait
            if CLICKBAIT_REGEX.search(t):
                clickbait_hits += 1
                
        # Normalisations
        vas_attention_ratio = attn_count / (n + 1e-5)
        vas_emotion_ratio   = emot_count / (n + 1e-5)
        
        # Index > 1.0 s'il y a plus d'attention que d'émotion
        vas_emoji_spam_idx  = (attn_count + 1.0) / (emot_count + 1.0)
        
        vas_all_caps_ratio  = all_caps_words / (total_words + 1e-5)
        vas_clickbait_ratio = clickbait_hits / (n + 1e-5)
        
        # Composite empirique : 
        # Les traits spam augmentent le score, l'émotion humaine le réduit fortement.
        spam_score = (
            vas_attention_ratio * 2.0 +
            vas_all_caps_ratio * 3.0 +
            vas_clickbait_ratio * 4.0 +
            np.log1p(vas_emoji_spam_idx) * 1.5
        ) - (vas_emotion_ratio * 2.0)
        
        records.append({
            "user_id": uid,
            "vas_attention_ratio": round(float(vas_attention_ratio), 4),
            "vas_emotion_ratio":   round(float(vas_emotion_ratio), 4),
            "vas_emoji_spam_idx":  round(float(vas_emoji_spam_idx), 4),
            "vas_all_caps_ratio":  round(float(vas_all_caps_ratio), 4),
            "vas_clickbait_ratio": round(float(vas_clickbait_ratio), 4),
            "vas_spam_score":      round(float(spam_score), 4)
        })
        
    vas_df = pd.DataFrame(records)
    if vas_df.empty:
        col_names = ["vas_attention_ratio", "vas_emotion_ratio", "vas_emoji_spam_idx",
                     "vas_all_caps_ratio", "vas_clickbait_ratio", "vas_spam_score"]
        vas_df = pd.DataFrame({"user_id": u_df["user_id"].astype(str)})
        for c in col_names:
            vas_df[c] = 0.0
        return vas_df

    # Remplir les manquants par défaut (0)
    vas_df["user_id"] = vas_df["user_id"].astype(str)
    all_users = pd.DataFrame({"user_id": u_df["user_id"].astype(str)})
    merged    = all_users.merge(vas_df, on="user_id", how="left").fillna(0.0)
    
    # Clippage de sécurité pour le composite score
    merged["vas_spam_score"] = merged["vas_spam_score"].clip(lower=0.0)
    
    return merged
