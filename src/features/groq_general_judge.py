import os
import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

class GroqGeneralJudge:
    def __init__(self, u_df: pd.DataFrame, p_df: pd.DataFrame, train_feat: pd.DataFrame, train_labels: pd.Series):
        """
        Unified LLM Judge that evaluates an ambiguous account by aggregating context, 
        model signals, and K-NN stylistic neighbors.
        """
        self.u_df = u_df
        self.p_df = p_df
        self.train_feat = train_feat
        self.train_labels = train_labels
        
        self.headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

    def _extract_top_features(self, uid: str, feat_df: pd.DataFrame) -> Tuple[str, str]:
        """Calculates Z-Scores to find the most anomalous features acting as Accusers/Protectors."""
        cols = [c for c in feat_df.columns if c not in ["is_bot", "account_id"]]
        if len(feat_df) < 5 or uid not in feat_df.index:
            return "Unknown", "Unknown"
            
        mean = feat_df[cols].mean()
        std = feat_df[cols].std().replace(0, 1e-9)
        z_scores = (feat_df.loc[uid, cols] - mean) / std
        
        # Sort features by highest absolute deviance
        sorted_feats = z_scores.abs().sort_values(ascending=False).head(3)
        top = [f"{f} (z={z_scores[f]:.2f})" for f in sorted_feats.index]
        return ", ".join(top), "General Temporal/Structural (Approximated)"

    def _find_nearest_exemplars(self, uid: str, feat_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cols = [c for c in feat_df.columns if c not in ["is_bot", "account_id"]]
        if uid not in feat_df.index:
            return [], []
            
        X_te = feat_df.loc[[uid], cols].fillna(0).values
        X_tr = self.train_feat[cols].fillna(0).values
        y_tr = self.train_labels.values
        
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        
        nn = NearestNeighbors(n_neighbors=20, metric="euclidean")
        nn.fit(X_tr_sc)
        distances, indices = nn.kneighbors(X_te_sc)
        
        bots_found = []
        humans_found = []
        
        for idx in indices[0]:
            is_bot = y_tr[idx]
            uid_train = self.train_feat.index[idx]
            if is_bot == 1 and len(bots_found) < 3:
                bots_found.append(uid_train)
            elif is_bot == 0 and len(humans_found) < 3:
                humans_found.append(uid_train)
            if len(bots_found) == 3 and len(humans_found) == 3:
                break
        return bots_found, humans_found

    def _format_posts(self, uid: str) -> str:
        if self.p_df is None: return "No tweets."
        up = self.p_df[self.p_df["user_id"] == uid]
        if up.empty: return "No tweets."
        texts = up["text"].fillna("").astype(str).tolist()
        texts = sorted(texts, key=len, reverse=True)[:10]
        return "\n".join([f"- {t}" for t in texts])

    def evaluate_account(self, uid: str,  lgbm_prob: float, court_votes: dict, 
                         miner_nomination: str, feat_df: pd.DataFrame, 
                         judge_mode: str = "GENERAL") -> Dict[str, Any]:
                             
        if not GROQ_API_KEY:
            return {"final_recommendation": "ABSTAIN", "reasoning_short": ["No API Key."]}

        # Profil
        user_row = self.u_df[self.u_df["user_id"] == uid]
        if user_row.empty:
            return {"final_recommendation": "ABSTAIN", "reasoning_short": ["User not found."]}
        user_row = user_row.iloc[0]
        
        # Meta
        b_votes = sum(1 for v in court_votes.values() if v["vote"] == "bot")
        h_votes = sum(1 for v in court_votes.values() if v["vote"] == "human")
        accusers, protectors = self._extract_top_features(uid, feat_df)
        b_ids, h_ids = self._find_nearest_exemplars(uid, feat_df)
        
        bot_neighbors_txt = "\n".join([f"Bot {i+1}: {self._format_posts(b)}" for i, b in enumerate(b_ids)])
        hum_neighbors_txt = "\n".join([f"Human {i+1}: {self._format_posts(h)}" for i, h in enumerate(h_ids)])
        
        # Lang/Event
        lang = "FR" if "FR_CLEAN_PERSONA" in judge_mode else "EN" if "EN_OVERPLAYED_HUMAN" in judge_mode else "UNKNOWN"
        
        user_prompt = f"""Analyze the following ambiguous account for bot detection.

[ACCOUNT]
language: {lang}
account_id: {uid}
username: {user_row.get('username', '')}
name: {user_row.get('name', '')}
description: {user_row.get('description', '')}
location: {user_row.get('location', '')}
tweet_count: {user_row.get('public_metrics.tweet_count', user_row.get('tweet_count', '?'))}

[MODEL SIGNALS]
model_probability_bot: {lgbm_prob:.3f}
court_votes_bot: {b_votes}
court_votes_human: {h_votes}
was_nominated: {miner_nomination}
top_accusers: {accusers}
top_protectors: {protectors}

[POSTS]
{self._format_posts(uid)}

[NEAREST BOT NEIGHBORS]
{bot_neighbors_txt}

[NEAREST HUMAN NEIGHBORS]
{hum_neighbors_txt}

Important:
- Focus on whether this account is genuinely human, genuinely bot, or too ambiguous.
- Distinguish 'performed humanity' from natural human messiness.
- Distinguish promo-masked bots from real community/fandom/quote accounts.
- Be conservative with false positives.
- Return STRICT JSON only.
"""

        system_prompt = """You are a highly conservative bot-detection adjudicator.

Your job is NOT to classify all accounts.
Your job is ONLY to analyze one ambiguous account at a time, using:
- account profile fields
- posts
- model signals
- court votes
- nearest bot neighbors
- nearest human neighbors

You must be extremely careful with false positives.
If evidence is mixed, choose ABSTAIN.
Never hallucinate hidden facts.
Use only the information provided.

You must return STRICT JSON only, with no markdown, no prose outside JSON.

Output schema:
{
  "bot_likelihood": 0.0,
  "human_likelihood": 0.0,
  "confidence": 0.0,
  "performed_human_persona": 0.0,
  "promotion_masked_as_human": 0.0,
  "community_human_account": 0.0,
  "roleplay_or_quote_human": 0.0,
  "reasoning_short": [
    "string",
    "string",
    "string"
  ],
  "final_recommendation": "BOT" | "HUMAN" | "ABSTAIN"
}

Rules:
- bot_likelihood + human_likelihood do not need to sum exactly to 1, but should be sensible.
- If unsure, output ABSTAIN.
- Do not over-penalize repetitive human fan accounts or quote/RP accounts without strong evidence.
- Distinguish between genuine human messiness and overperformed human imitation.
"""

        # Modes conditionnels
        if judge_mode == "EN_OVERPLAYED_HUMAN":
            user_prompt += """
[SPECIAL MODE: EN_OVERPLAYED_HUMAN]
Goal: detect EN bots that overplay humanity without condemning atypical humans.
Pay special attention to:
- too generic but attractive lifestyle bio
- mistakes that are 'too well distributed'
- extremely natural hashtags but repetitive
- well-packaged personal posts
- soft but over-performed human persona
- fluid narrative + artificial imperfections
Provide strict focus on performed_human_persona and community_human_account.
"""
        elif judge_mode == "FR_CLEAN_PERSONA":
             user_prompt += """
[SPECIAL MODE: FR_CLEAN_PERSONA]
Goal: detect highly clean, conversational FR bots without breaking FR humans.
Pay special attention to:
- too credible/smooth French persona
- clean tone, almost zero human noise
- total absence of linguistic relaxation
- promotion hidden under a personal voice
- argument structures too well formed
- artificial FR/EN mixing
- generic but perfectly compatible bot bio
Provide strict focus on promotion_masked_as_human and performed_human_persona.
"""

        payload = {
            "model": GROQ_MODEL,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }
        
        try:
            r = requests.post(GROQ_ENDPOINT, headers=self.headers, json=payload, timeout=30)
            data = r.json()
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception as e:
            return {"final_recommendation": "ABSTAIN", "reasoning_short": [f"Error: {e}"]}
