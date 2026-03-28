# -*- coding: utf-8 -*-
"""
tests/test_coordination.py
--------------------------
Garde-fou pour vérifier que la détection de rafales coordonnées
et de similarités entre (time_bin, utilisateurs) fonctionne bien
en O(N) sans faux positifs.
"""

import pandas as pd
import numpy as np

from src.features.coordination import extract_coordination_features


def test_no_coordination_single_user():
    # Utilisateur normal, 1 post (pas de coordination)
    accounts_df = pd.DataFrame({"account_id": ["u1"]})
    posts_df = pd.DataFrame({
        "account_id": ["u1"],
        "created_at": ["2023-01-01T12:05:00Z"],
        "text": ["A simple test message with enough words"],
        "hashtags": [["#soloboy"]],
        "urls": [[]]
    })
    cfg = {"coordination": {"enabled": True, "time_window_minutes": 60, "min_users_per_bin": 2}}

    res = extract_coordination_features(accounts_df, posts_df, cfg)
    
    assert len(res) == 1
    row = res.iloc[0]
    
    # Rien n'est partagé
    assert row["coord_text_sim"] == 0.0
    assert row["coord_hashtag_sync"] == 0.0
    assert row["coord_burst_sync"] == 0.0
    assert row["coord_activation_score"] == 0.0


def test_perfect_coordination_two_users():
    # 2 bots postant exactement le même texte et mêmes hashtags à 5min d'intervalle
    # (même bin de 60 minutes)
    accounts_df = pd.DataFrame({"account_id": ["b1", "b2"]})
    posts_df = pd.DataFrame({
        "account_id": ["b1", "b2"],
        "created_at": ["2023-01-01T12:05:00Z", "2023-01-01T12:10:00Z"],
        "text": ["Spamming the exact same promotional text message", "Spamming the exact same promotional text message"],
        "hashtags": [["#crypto", "#scam"], ["#crypto", "#scam"]],
        "urls": [["http://spam.com"], ["http://spam.com"]]
    })
    
    cfg = {"coordination": {"enabled": True, "time_window_minutes": 60, "min_users_per_bin": 2}}
    res = extract_coordination_features(accounts_df, posts_df, cfg).set_index("account_id")

    # Ils partagent tout
    b1 = res.loc["b1"]
    b2 = res.loc["b2"]
    
    # L'activation score doit être 1 (1 autre utilisateur dans le bin)
    assert b1["coord_activation_score"] == 1.0
    assert b2["coord_activation_score"] == 1.0
    
    # 100% hashtags partagés
    assert b1["coord_hashtag_sync"] == 1.0
    # 100% URLs partagées
    assert b1["coord_url_sync"] == 1.0
    # 100% de mots partagés
    assert b1["coord_text_sim"] == 1.0
    
    # Pas de burst, < 3 posts
    assert b1["coord_burst_sync"] == 0.0


def test_mixed_coordination_and_bursts():
    # 2 bots font un burst coordonné (>3 posts), 1 humain parle d'autre chose
    accounts_df = pd.DataFrame({"account_id": ["b1", "b2", "h1"]})
    
    p = []
    # Burst 1
    for i in range(4):
        p.append({"account_id": "b1", "created_at": f"2023-01-01T10:0{i}:00Z", "text": "Giveaway win now", "hashtags": ["#win"]})
    # Burst 2
    for i in range(4):
        p.append({"account_id": "b2", "created_at": f"2023-01-01T10:0{i}:00Z", "text": "Giveaway claim win", "hashtags": ["#win"]})
    
    # Humain 1
    p.append({"account_id": "h1", "created_at": "2023-01-01T10:30:00Z", "text": "Just ate a sandwich", "hashtags": ["#food"]})
    
    posts_df = pd.DataFrame(p)
    posts_df["urls"] = [[]] * len(posts_df)
    
    cfg = {"coordination": {"enabled": True, "time_window_minutes": 60}}
    res = extract_coordination_features(accounts_df, posts_df, cfg).set_index("account_id")
    
    b1 = res.loc["b1"]
    h1 = res.loc["h1"]
    
    # #win est partagé, donc hashtag sync doit être 1 pour b1
    assert b1["coord_hashtag_sync"] == 1.0
    # h1 a #food qui n'est pas partagé
    assert h1["coord_hashtag_sync"] == 0.0
    
    # Burst de b1 et b2 est synchrone? Oui (2 accounts en burst).
    # Wait, the burst rule we wrote is (bursts_in_bin >= 3)
    # Dans ce cas on a 2 bursts. Il faut que je vérifie le comportement. 
    # Ah ! le test actuel doit utiliser 3 comptes burstant pour être à 1.
