import pandas as pd
import numpy as np

def extract_time_delta_v2(u_df, p_df):
    results = []
    for uid, g in p_df.groupby("user_id"):
        times = pd.to_datetime(g.get("created_at", pd.Series()), errors='coerce').dropna().sort_values()
        if len(times) < 2:
            results.append({"user_id": uid, "delta_v2_mean": 0.0, "delta_v2_std": 0.0})
            continue
        deltas = times.diff().dt.total_seconds().dropna()
        results.append({
            "user_id": uid,
            "delta_v2_mean": float(deltas.mean()),
            "delta_v2_std": float(deltas.std()) if len(deltas)>1 else 0.0
        })
    return pd.DataFrame(results)
