# -*- coding: utf-8 -*-
"""
tests/test_historical_adapter.py
"""

import json
import pytest
from pathlib import Path
from src.data.adapters.historical_adapter import HistoricalAdapter
from src.data.schema import AccountCols, PostCols, LabelCols

def test_historical_adapter_loading(tmp_path):
    # Setup un mock dataset historique
    mock_json = {
        "users": [
            {
                "id": "u1", "name": "Alice", "username": "alice99",
                "tweet_count": 50, "description": "Hello", "location": "Paris"
            },
            {
                "id": "u2", "name": "Bot1", "username": "bot_one",
                "tweet_count": 5000, "description": "Spam", "location": "Web"
            }
        ],
        "posts": [
            {
                "id": "p1", "author_id": "u1", "text": "Hi", "created_at": "2023-01-01T10:00:00Z", "lang": "en", "source": "Web"
            },
            {
                "id": "p2", "author_id": "u2", "text": "Buy now", "created_at": "2023-01-01T10:01:00Z", "lang": "en", "source": "API"
            }
        ]
    }
    
    mock_txt = "u2\nmissing_u3\n"
    
    json_path = tmp_path / "dataset.posts&users.99.json"
    txt_path = tmp_path / "dataset.bots.99.txt"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mock_json, f)
        
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(mock_txt)
        
    adapter = HistoricalAdapter()
    
    # Test can_handle
    assert adapter.can_handle(json_path) is True
    assert adapter.can_handle(txt_path) is False
    
    # Test load
    bundle = adapter.load(json_path)
    
    assert bundle.n_accounts == 2
    assert bundle.n_posts == 2
    assert bundle.source_format == "historical"
    
    # Verify Accounts
    act = bundle.accounts_df
    assert AccountCols.ID in act.columns
    assert AccountCols.SCREEN_NAME in act.columns
    assert AccountCols.TOTAL_POSTS in act.columns
    assert len(act) == 2
    assert act.loc[act[AccountCols.ID] == "u1", AccountCols.SCREEN_NAME].iloc[0] == "alice99"
    
    # Verify Posts
    pst = bundle.posts_df
    assert PostCols.ID in pst.columns
    assert PostCols.ACCOUNT_ID in pst.columns
    assert PostCols.TEXT in pst.columns
    
    # Verify Labels
    lbl = bundle.labels_df
    assert lbl is not None
    assert len(lbl) == 2
    assert LabelCols.LABEL in lbl.columns
    assert lbl.loc[lbl[LabelCols.ACCOUNT_ID] == "u1", LabelCols.LABEL].iloc[0] == 0.0 # Humain
    assert lbl.loc[lbl[LabelCols.ACCOUNT_ID] == "u2", LabelCols.LABEL].iloc[0] == 1.0 # Bot
