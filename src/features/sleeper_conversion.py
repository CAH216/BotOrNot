# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_sleeper_conversion(u_df: pd.DataFrame, p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les comptes recyclés (Sleeper Conversion).
    Mesure le mismatch sémantique (via Cosine Sim TF-IDF) entre la Bio originelle et les textes récents.
    Cible les profils = 1 à 3 posts qui présentent une cassure de style descriptif.
    """
    n = len(u_df)
    res = pd.DataFrame({"user_id": u_df["user_id"].unique()})
    
    expected_cols = ["sleeper_bio_post_mismatch", "sleeper_abrupt_shift_2_posts"]
    for c in expected_cols: res[c] = 0.0
    
    if n == 0 or p_df.empty or "text" not in p_df.columns:
        return res
        
    p_grouped = p_df.groupby("user_id")["text"].apply(list).to_dict()
    
    mismatches = np.zeros(n)
    abrupt_shifts = np.zeros(n)
    
    # Pre-train a tiny TF-IDF on all texts just to map the vocabulary
    all_texts = (u_df.get("description", pd.Series(dtype=str)).fillna("").tolist() + 
                 p_df["text"].fillna("").tolist())
    tfidf = TfidfVectorizer(max_features=1000, analyzer="char_wb", ngram_range=(3, 5))
    try:
        tfidf.fit(all_texts)
    except Exception:
        return res
    
    for i, row in u_df.iterrows():
        uid = row["user_id"]
        bio = str(row.get("description", ""))
        posts = p_grouped.get(uid, [])
        
        if len(bio) < 5 or len(posts) == 0:
            continue
            
        posts_text = " ".join([str(t) for t in posts])
        
        if len(posts_text) < 5:
            continue
            
        bio_vec = tfidf.transform([bio.lower()])
        posts_vec = tfidf.transform([posts_text.lower()])
        
        sim = cosine_similarity(bio_vec, posts_vec)[0][0]
        # Mismatch is higher when similarity is lower
        mismatches[i] = 1.0 - sim
        
        # Shift on 2 posts exactly
        if len(posts) > 1:
            p1_vec = tfidf.transform([str(posts[0]).lower()])
            p2_vec = tfidf.transform([str(posts[-1]).lower()])
            sim_p1_p2 = cosine_similarity(p1_vec, p2_vec)[0][0]
            abrupt_shifts[i] = 1.0 - sim_p1_p2
            
    res["sleeper_bio_post_mismatch"] = mismatches
    res["sleeper_abrupt_shift_2_posts"] = abrupt_shifts
    
    return res
