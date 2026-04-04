# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def extract_account_similarity_graph(u_df: pd.DataFrame, p_df: pd.DataFrame, y=None) -> pd.DataFrame:
    """
    Extrait les propriétés locales du graphe de similarité (Mission 2).
    Si y est fourni (pendant l'entraînement), calcule la densité locale de bots.
    Sinon, extrait la densité de la communauté (clones sémantiques).
    """
    n = len(u_df)
    res = pd.DataFrame({"user_id": u_df["user_id"].unique()})
    
    expected_cols = ["graph_community_size", "graph_local_density", "graph_bot_neighbor_ratio"]
    for c in expected_cols: res[c] = 0.0
    
    if n <= 1:
        return res
        
    texts = (u_df.get("username", pd.Series(dtype=str)).fillna("") + " " +
             u_df.get("name", pd.Series(dtype=str)).fillna("") + " " +
             u_df.get("description", pd.Series(dtype=str)).fillna("") + " " +
             u_df.get("location", pd.Series(dtype=str)).fillna("")).str.lower()
             
    tfidf = TfidfVectorizer(max_features=2000, analyzer="char_wb", ngram_range=(3,5)) # Character n-grams for typo skeletons
    X = tfidf.fit_transform(texts)
    
    # Sim matrix (sparse matrix multiplication is fast)
    S = cosine_similarity(X, X)
    
    com_sizes = np.zeros(n)
    local_den = np.zeros(n)
    bot_neigh = np.zeros(n)
    
    # Threshold for community edge
    edge_thresh = 0.85
    
    for i in range(n):
        sims = S[i]
        # Ignore self
        sims[i] = 0.0
        
        edges = sims > edge_thresh
        n_edges = edges.sum()
        
        com_sizes[i] = n_edges
        local_den[i] = sims.mean() # Average similarity across entire dataset
        
        if y is not None and n_edges > 0:
            # How many neighbors are actually labelled bot? (if training data is known)
            bot_neigh[i] = y[edges].sum() / n_edges
            
    res["graph_community_size"] = com_sizes
    res["graph_local_density"] = local_den
    res["graph_bot_neighbor_ratio"] = bot_neigh
    
    return res
