import pandas as pd
import numpy as np
import re
from collections import Counter

def extract_llm_hallucinations(u_df: pd.DataFrame, p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les signaux trahissant une IA générative mal paramétrée :
    - Caractères de contrôle invalides (mojibake)
    - Homogénéité suspecte de la fin des tweets (Suffixe forcé)
    """
    feats = []
    
    # Regex précompilées
    re_control = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

    for uid, g in p_df.groupby("user_id"):
        texts = g["text"].fillna("").astype(str).tolist()
        
        control_count = 0
        suffixes = []

        for t in texts:
            # 1. Contrôle & Échappements plantés
            control_count += len(re_control.findall(t))
            control_count += t.count('\\u000') + t.count('\\b') + t.count('\\f')
            
            # 2. Suffixe (ex. "LOL...")
            # On retire d'abord "twitter_link" (artefact de preprocessing)
            clean_t = t.replace("https://t.co/twitter_link", "").replace("twitter_link", "").strip()
            if len(clean_t) >= 6:
                suffixes.append(clean_t[-6:])
        
        # Calcul Homogénéité
        suffix_homogeneity = 0.0
        if len(suffixes) >= 3:
            c = Counter(suffixes)
            most_common = c.most_common(1)[0][1]
            suffix_homogeneity = most_common / len(suffixes)

        feats.append({
            "user_id": uid,
            "llm_control_chars_count": float(control_count),
            "llm_internal_caps_count": 0.0,
            "llm_suffix_homogeneity": float(suffix_homogeneity)
        })
    
    df = pd.DataFrame(feats)
    
    # Jointure pour s'assurer que tous les users de u_df sont présents
    all_uids = pd.DataFrame({"user_id": u_df["user_id"].unique()})
    if not df.empty:
        df = all_uids.merge(df, on="user_id", how="left").fillna(0.0)
    else:
        df = all_uids.copy()
        df["llm_control_chars_count"] = 0.0
        df["llm_internal_caps_count"] = 0.0
        df["llm_suffix_homogeneity"] = 0.0
        
    return df
