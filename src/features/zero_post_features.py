# -*- coding: utf-8 -*-
"""
zero_post_features.py
Extracteur de variables statiques ciblant spécifiquement les comptes avec très peu de données.
"""
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.ensemble import IsolationForest

def extract_zero_post_features(users_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"user_id": users_df["user_id"].unique()})
    
    # Fill defaults
    expected_cols = [
        "zp_name_len", "zp_screen_name_len", "zp_bio_len", "zp_loc_len",
        "zp_screen_name_digits_ratio", "zp_screen_name_entropy", "zp_name_sim",
        "zp_bio_empty", "zp_loc_empty", "zp_bio_clone_score", "zp_sn_pattern_freq",
        "zp_iso_forest_score", "zp_favs_per_day", "zp_followers_per_day"
    ]
    for c in expected_cols: df[c] = 0.0
    
    if len(users_df) == 0:
        return df
        
    res = []
    
    # Pre-computation for Cross-Account (Family C)
    bios = users_df["description"].fillna("").astype(str).str.strip().str.lower()
    bio_counts = Counter([b for b in bios if len(b) > 0])
    
    def _sn_skel(s):
        s = str(s)
        s = re.sub(r'[A-Za-z]', 'w', s)
        s = re.sub(r'[0-9]', 'N', s)
        return s
        
    sn_skels = users_df.get("username", pd.Series(dtype=str)).fillna("").apply(_sn_skel)
    sn_skel_counts = Counter(sn_skels)
    
    n_users = max(len(users_df), 1)
    
    now = pd.to_datetime("2024-01-01", utc=True) # Static reference pour consistance
    
    # Matrices pour Family D IsolationForest
    tab_data = [] # pour entrainer l'iso forest
    
    for _, row in users_df.iterrows():
        uid = row["user_id"]
        
        # Famille A: Morphology
        sn = str(row.get("username", ""))
        nm = str(row.get("name", ""))
        bio = str(row.get("description", ""))
        loc = str(row.get("location", ""))
        
        zp_name_len = len(nm)
        zp_screen_name_len = len(sn)
        zp_bio_len = len(bio)
        zp_loc_len = len(loc)
        
        sn_digits = len(re.findall(r'[0-9]', sn))
        zp_screen_name_digits_ratio = sn_digits / max(len(sn), 1)
        
        char_counts = Counter(sn).values()
        probs = [c / max(len(sn), 1) for c in char_counts]
        zp_screen_name_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        s1, s2 = set(sn.lower()), set(nm.lower())
        zp_name_sim = len(s1.intersection(s2)) / max(len(s1.union(s2)), 1)
        
        # Famille B: Persona Template
        zp_bio_empty = 1.0 if not bio.strip() else 0.0
        zp_loc_empty = 1.0 if not loc.strip() else 0.0
        
        # Famille C: Clones
        b_low = bio.strip().lower()
        zp_bio_clone_score = bio_counts.get(b_low, 0) / n_users if b_low else 0.0
        
        skel = _sn_skel(sn)
        zp_sn_pattern_freq = sn_skel_counts.get(skel, 0) / n_users
        
        # Famille D (Pre-computation)
        created_at = pd.to_datetime(row.get("created_at"), errors="coerce", utc=True)
        age_days = (now - created_at).days if pd.notna(created_at) else 365
        age_days = max(age_days, 1)
        
        favs = float(row.get("favourites_count", 0))
        fols = float(row.get("followers_count", 0))
        stats = float(row.get("statuses_count", 0))
        frie = float(row.get("friends_count", 0))
        
        zp_favs_per_day = favs / age_days
        zp_followers_per_day = fols / age_days
        
        tab_data.append([favs, fols, stats, frie])
        
        res.append({
            "user_id": uid,
            "zp_name_len": zp_name_len,
            "zp_screen_name_len": zp_screen_name_len,
            "zp_bio_len": zp_bio_len,
            "zp_loc_len": zp_loc_len,
            "zp_screen_name_digits_ratio": zp_screen_name_digits_ratio,
            "zp_screen_name_entropy": zp_screen_name_entropy,
            "zp_name_sim": zp_name_sim,
            "zp_bio_empty": zp_bio_empty,
            "zp_loc_empty": zp_loc_empty,
            "zp_bio_clone_score": zp_bio_clone_score,
            "zp_sn_pattern_freq": zp_sn_pattern_freq,
            "zp_favs_per_day": zp_favs_per_day,
            "zp_followers_per_day": zp_followers_per_day
        })
        
    res_df = pd.DataFrame(res)
    
    # 4. Family D Isolation Forest Score
    # Fit en temps réel sur le batch ou le dataset
    X_iso = np.nan_to_num(np.array(tab_data))
    if len(X_iso) > 10:
        iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        # score_samples returns opposite of anomaly score. Smaller = more anomalous.
        # We invert it so higher = anomalous (Bot)
        iso.fit(X_iso)
        scores = -iso.score_samples(X_iso)
        res_df["zp_iso_forest_score"] = scores
    else:
        res_df["zp_iso_forest_score"] = 0.0
        
    # Replace the empty columns in df with our computed features
    for c in expected_cols:
        if c in df.columns: df = df.drop(columns=[c])
        
    return df.merge(res_df, on="user_id", how="left").fillna(0.0)
