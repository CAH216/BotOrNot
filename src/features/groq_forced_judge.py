import os
import json
import requests
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

class GroqForcedJudge:
    def __init__(self, u_df: pd.DataFrame, p_df: pd.DataFrame, train_feat: pd.DataFrame, train_labels: pd.Series):
        self.u_df = u_df
        self.p_df = p_df
        self.train_feat = train_feat
        self.train_labels = train_labels
        self.headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

    def _format_posts(self, uid: str) -> str:
        if self.p_df is None: return "No tweets."
        up = self.p_df[self.p_df["user_id"] == uid]
        if up.empty: return "No tweets."
        texts = up["text"].fillna("").astype(str).tolist()
        texts = sorted(texts, key=len, reverse=True)[:10]
        return "\n".join([f"- {t}" for t in texts])

    def _call_groq(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not GROQ_API_KEY:
            return {"verdict": "ABSTAIN", "reasoning_short": ["No API Key."]}
            
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
            return {"verdict": "ABSTAIN", "reasoning_short": [f"Error: {e}"]}

    def _build_context(self, uid: str,  lgbm_prob: float, court_votes: dict, feat_df: pd.DataFrame) -> str:
        user_row = self.u_df[self.u_df["user_id"] == uid]
        if user_row.empty: return ""
        user_row = user_row.iloc[0]
        
        b_votes = sum(1 for v in court_votes.values() if v["vote"] == "bot")
        h_votes = sum(1 for v in court_votes.values() if v["vote"] == "human")
        
        context = f"""
[ACCOUNT PROFILE]
account_id: {uid}
username: {user_row.get('username', '')}
description: {user_row.get('description', '')}

[MODEL SIGNALS]
machine_learning_bot_probability: {lgbm_prob:.3f}
court_votes_bot: {b_votes}
court_votes_human: {h_votes}

[POSTS]
{self._format_posts(uid)}
"""
        return context

    def evaluate_fp_veto(self, uid: str, lgbm_prob: float, court_votes: dict, feat_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Usage: False Positive potential.
        Model=BOT, Court=HUMAN.
        """
        context = self._build_context(uid, lgbm_prob, court_votes, feat_df)
        if not context: return {"verdict": "ABSTAIN"}
        
        system_prompt = """You are a highly analytical bot-detection adjudicator evaluating a POTENTIAL FALSE POSITIVE.
An automated ML model flagged this target as a BOT. 
However, independent analytical courts suspect this might be a genuine 'Atypical Human' (e.g., intense sports fan, roleplayer, dedicated community member, quote poster).

Your job: Determine definitively if the account is an atypical HUMAN, or just a BOT.

You MUST choose exactly 'HUMAN' or 'BOT'. 
You CANNOT choose ABSTAIN.
Provide your reasoning, your confidence level (0.0 to 1.0), and the margin (0.0 to 1.0 representing how far apart your two choices were).

Output strict JSON only:
{
  "verdict": "HUMAN" | "BOT",
  "confidence": 0.0,
  "margin": 0.0,
  "reasoning_short": ["..."],
  "risk_if_wrong": "low|medium|high"
}
"""
        return self._call_groq(system_prompt, context)

    def evaluate_fn_rescue(self, uid: str, lgbm_prob: float, court_votes: dict, feat_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Usage: False Negative potential.
        Model=HUMAN, Court=BOT.
        """
        context = self._build_context(uid, lgbm_prob, court_votes, feat_df)
        if not context: return {"verdict": "ABSTAIN"}
        
        system_prompt = """You are a highly analytical bot-detection adjudicator evaluating a POTENTIAL FALSE NEGATIVE.
An automated ML model flagged this target as a HUMAN because it's clean and lacks obvious temporal spam signals.
However, independent analytical courts suspect this might be a sophisticated disguised BOT (e.g., LLM Persona, clean promo bot, masked behavior).

Your job: Determine definitively if the account is a disguised BOT, or just a genuine boring HUMAN.

You MUST choose exactly 'BOT' or 'HUMAN'. 
You CANNOT choose ABSTAIN.
Provide your reasoning, your confidence level (0.0 to 1.0), and the margin (0.0 to 1.0 representing how far apart your two choices were).

Output strict JSON only:
{
  "verdict": "HUMAN" | "BOT",
  "confidence": 0.0,
  "margin": 0.0,
  "reasoning_short": ["..."],
  "risk_if_wrong": "low|medium|high"
}
"""
        return self._call_groq(system_prompt, context)
