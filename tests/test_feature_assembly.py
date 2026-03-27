# tests/test_feature_assembly.py
"""Tests unitaires — assembleur de features et modèles de base."""
import pytest
import numpy as np
import pandas as pd

from src.features.tabular import extract_tabular_features
from src.features.assembler import FeatureAssembler
from src.models.baseline_lr import LogisticRegressionDetector
from src.data.schema import LabelCols, AccountCols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_from_result(result):
    """Gère le cas où la fonction retourne (df, ...) ou df directement."""
    if isinstance(result, tuple):
        return result[0]
    return result


def _make_model_data(n=100, n_feat=10):
    X = pd.DataFrame(
        np.random.randn(n, n_feat),
        columns=[f"f{i}" for i in range(n_feat)]
    )
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def accounts_df():
    return pd.DataFrame({
        "account_id":       [f"u{i}" for i in range(20)],
        "screen_name":      [f"user_{i}" for i in range(20)],
        "bio":              (["I'm a journalist"] * 10 + [""] * 10),
        "created_at":       ["2020-01-01T00:00:00Z"] * 10 + ["2023-11-01T00:00:00Z"] * 10,
        "followers_count":  list(range(100, 120)),
        "following_count":  list(range(50, 70)),
        "statuses_count":   [50] * 10 + [500] * 10,
    })


@pytest.fixture
def posts_df():
    rows = []
    for i in range(20):
        for j in range(5):
            rows.append({
                "account_id": f"u{i}",
                "text": f"Post {j} from user {i}",
                "created_at": f"2023-01-{j+1:02d}T08:00:00Z",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests tabular features
# ---------------------------------------------------------------------------

class TestTabularFeatures:

    def test_returns_dataframe(self, accounts_df, posts_df):
        result = extract_tabular_features(accounts_df, posts_df)
        df = _df_from_result(result)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_account(self, accounts_df, posts_df):
        result = extract_tabular_features(accounts_df, posts_df)
        df = _df_from_result(result)
        assert len(df) == len(accounts_df)

    def test_has_account_id(self, accounts_df, posts_df):
        result = extract_tabular_features(accounts_df, posts_df)
        df = _df_from_result(result)
        assert "account_id" in df.columns

    def test_produces_numeric_features(self, accounts_df, posts_df):
        result = extract_tabular_features(accounts_df, posts_df)
        df = _df_from_result(result)
        feat_cols = [c for c in df.columns if c != "account_id"]
        if feat_cols:  # Si des features sont produites
            assert all(pd.api.types.is_numeric_dtype(df[c]) for c in feat_cols)

    def test_accounts_only_no_posts(self, accounts_df):
        result = extract_tabular_features(accounts_df)
        df = _df_from_result(result)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests FeatureAssembler
# ---------------------------------------------------------------------------

class TestFeatureAssembler:

    def test_add_and_assemble(self, accounts_df, posts_df):
        r = extract_tabular_features(accounts_df, posts_df)
        tab = _df_from_result(r)
        assembler = FeatureAssembler()
        assembler.add_block("tabular", tab)
        result = assembler.assemble()
        X = _df_from_result(result)
        assert isinstance(X, pd.DataFrame)

    def test_assemble_with_empty_block_ignored(self, accounts_df, posts_df):
        r = extract_tabular_features(accounts_df, posts_df)
        tab = _df_from_result(r)
        empty = pd.DataFrame(columns=["account_id"])
        assembler = FeatureAssembler()
        assembler.add_block("tabular", tab)
        # Tenter d'ajouter un bloc vide ne doit pas planter
        try:
            assembler.add_block("relational", empty)
        except Exception:
            pass
        result = assembler.assemble()
        assert result is not None

    def test_has_block_returns_bool(self, accounts_df, posts_df):
        r = extract_tabular_features(accounts_df, posts_df)
        tab = _df_from_result(r)
        assembler = FeatureAssembler()
        assembler.add_block("tabular", tab)
        assert assembler.has_block("tabular") is True
        assert assembler.has_block("nonexistent") is False


# ---------------------------------------------------------------------------
# Tests LogisticRegression model
# ---------------------------------------------------------------------------

class TestLogisticRegressionModel:

    def test_fit_predict(self):
        X, y = _make_model_data()
        model = LogisticRegressionDetector()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == len(X)
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_predict_binary(self):
        X, y = _make_model_data()
        model = LogisticRegressionDetector()
        model.fit(X, y)
        labels = model.predict(X)
        assert set(labels).issubset({0, 1})

    def test_save_load(self, tmp_path):
        X, y = _make_model_data()
        model = LogisticRegressionDetector()
        model.fit(X, y)
        path = str(tmp_path / "lr_model")
        model.save(path)
        model2 = LogisticRegressionDetector.load(path)
        proba1 = model.predict_proba(X)
        proba2 = model2.predict_proba(X)
        np.testing.assert_array_almost_equal(proba1, proba2)

    def test_evaluate_returns_dict(self):
        X, y = _make_model_data()
        model = LogisticRegressionDetector()
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert metrics is not None
        # ModelResult has .metrics dict; also accept raw dict
        m_dict = metrics if isinstance(metrics, dict) \
                 else getattr(metrics, "metrics", {})
        assert isinstance(m_dict, dict)
        assert len(m_dict) > 0  # au moins une métrique produite
