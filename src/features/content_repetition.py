import pandas as pd
import numpy as np

def extract_content_repetition(u_df, p_df):
    results = []
    for uid, g in p_df.groupby("user_id"):
        texts = g["text"].fillna("").astype(str).tolist()
        if len(texts) < 2:
            results.append({
                "user_id": uid, "csr_jaccard_mean": 0.0, "csr_jaccard_std": 0.0,
                "csr_jaccard_min": 0.0, "csr_unigram_pct_shared": 0.0, "csr_template_score": 0.0
            })
            continue
        
        jaccards = []
        all_unigrams = []
        for t in texts:
            tokens = set(t.lower().split())
            all_unigrams.append(tokens)
            
        for i in range(len(all_unigrams)-1):
            s1 = all_unigrams[i]
            s2 = all_unigrams[i+1]
            if len(s1|s2) == 0:
                jaccards.append(1.0)
            else:
                jaccards.append(len(s1&s2) / len(s1|s2))
                
        # Intersection on all posts
        if len(all_unigrams) > 0:
            common = set.intersection(*all_unigrams)
            union = set.union(*all_unigrams)
            pct = len(common)/len(union) if len(union) > 0 else 0.0
        else:
            pct = 0.0
            
        jm = float(np.mean(jaccards))
        jstd = float(np.std(jaccards))
        jmin = float(np.min(jaccards))
        
        results.append({
            "user_id": uid,
            "csr_jaccard_mean": jm,
            "csr_jaccard_std": jstd,
            "csr_jaccard_min": jmin,
            "csr_unigram_pct_shared": float(pct),
            "csr_template_score": jm + pct
        })
    return pd.DataFrame(results)
