"""Script de génération du dataset dry-run."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300
rows = []
for i in range(n):
    is_bot = int(i < 150)
    followers = int(rng.integers(0, 200)) if is_bot else int(rng.exponential(1000))
    following = int(rng.integers(3000, 8000)) if is_bot else int(rng.integers(50, 800))
    statuses  = int(rng.integers(20000, 80000)) if is_bot else int(rng.exponential(5000))
    ts = pd.Timestamp("2023-01-01 02:00:00", tz="UTC") if is_bot else pd.Timestamp("2023-01-01 09:00:00", tz="UTC")
    for _ in range(10):
        gap = float(rng.uniform(2, 10)) if is_bot else float(rng.uniform(600, 7200))
        ts  = ts + pd.Timedelta(seconds=gap)
        text = "buy now! http://spam.com #promo" if is_bot else "Good morning everyone!"
        rows.append({
            "user_id": f"u{i}", "is_bot": is_bot,
            "followers_count": followers, "following_count": following,
            "statuses_count": statuses, "verified": False,
            "created_at": str(ts), "text": text,
        })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)

uids = df["user_id"].unique()
test_uids = set(uids[240:])
df[~df["user_id"].isin(test_uids)].to_csv("data/_dryrun_train.csv", index=False)
df[df["user_id"].isin(test_uids)].to_csv("data/_dryrun_test.csv", index=False)

n_train = len(df[~df["user_id"].isin(test_uids)]["user_id"].unique())
n_test  = len(test_uids)
print(f"Dataset dry-run : {n_train} comptes train / {n_test} comptes test")
