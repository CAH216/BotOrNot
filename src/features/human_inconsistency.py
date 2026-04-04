# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter

def extract_human_inconsistency(users_df: pd.DataFrame, posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simule l'inconsistance naturelle d'un humain.
    Alternance de longueurs extrêmes, entropie lexicale globale vs locale, etc.
    """
    df = pd.DataFrame({"user_id": users_df["user_id"].unique()})
    for c in ["human_length_ratio", "human_lexical_entropy"]: df[c] = 0.0
    
    if posts_df.empty or "text" not in posts_df.columns:
        return df
        
    def _inconsistency_vars(texts):
        if len(texts) < 2: return 0, 0
        
        # 1. Longueur alternance: variance de ratio entre log des posts min et max
        lengths = [len(str(t)) for t in texts if pd.notna(t)]
        if not lengths: return 0, 0
        l_min, l_max = min(lengths), max(lengths)
        ratio_extreme = l_max / max(l_min, 1)
        
        # 2. Entropie lexicale basique d'un utilisateur
        all_words = " ".join([str(t).lower() for t in texts if pd.notna(t)]).split()
        if not all_words: return ratio_extreme, 0
        
        counts = list(Counter(all_words).values())
        total = sum(counts)
        probs = [c / total for c in counts]
        entropy = -sum(p * np.log2(p) for p in probs)
        
        return ratio_extreme, entropy
        
    res = []
    grouped = posts_df.groupby("user_id")["text"].apply(list)
    
    for uid in users_df["user_id"]:
        texts = grouped.get(uid, [])
        r_ext, ent = _inconsistency_vars(texts)
        res.append({
            "user_id": uid,
            "human_length_ratio": np.log1p(r_ext), # log for safety
            "human_lexical_entropy": ent
        })
        
    return df.merge(pd.DataFrame(res), on="user_id", how="left").fillna(0)
