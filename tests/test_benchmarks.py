# -*- coding: utf-8 -*-
import os
import pytest
from src.data.loaders import load_bundle

@pytest.mark.skipif(not os.path.exists("dataset/dataset.posts&users.30.json"), reason="Historical event 30 not found")
def test_historical_event_30_loads():
    bundle = load_bundle("dataset/dataset.posts&users.30.json", adapter="historical")
    assert bundle.n_accounts > 0
    assert bundle.n_posts > 0
    assert bundle.labels_df is not None

@pytest.mark.skipif(not os.path.exists("dataset/dataset.posts&users.31.json"), reason="Historical event 31 not found")
def test_historical_event_31_loads():
    bundle = load_bundle("dataset/dataset.posts&users.31.json", adapter="historical")
    assert bundle.n_accounts > 0
    assert bundle.n_posts > 0
    assert bundle.labels_df is not None
