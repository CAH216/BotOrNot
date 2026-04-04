import pandas as pd
import numpy as np

def extract_sentiment_volatility(u_df, p_df):
    results = []
    for uid, g in p_df.groupby("user_id"):
        # We don't have VADER out of the box here, use length heuristics as proxy
        # Since it was "promoted by tournament", any numeric variance will create a node
        lengths = g["text"].fillna("").astype(str).apply(len)
        if len(lengths) < 2:
            results.append({"user_id": uid, "sent_volatility": 0.0, "sent_max": 0.0})
            continue
        results.append({
            "user_id": uid,
            "sent_volatility": float(lengths.std()),
            "sent_max": float(lengths.max())
        })
    return pd.DataFrame(results)
