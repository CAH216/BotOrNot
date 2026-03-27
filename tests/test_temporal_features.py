# tests/test_temporal_features.py
"""Tests unitaires — extraction de features temporelles."""
import pytest
import pandas as pd
import numpy as np

from src.features.temporal import extract_temporal_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def posts_precise():
    """Posts avec timestamps précis — Plan A."""
    return pd.DataFrame({
        "account_id": ["u1"] * 10 + ["u2"] * 10,
        "created_at": (
            pd.date_range("2023-01-01 08:00", periods=10, freq="5min").tolist()
            + pd.date_range("2023-01-02 22:00", periods=10, freq="1min").tolist()
        ),
    })


@pytest.fixture
def posts_date_only():
    """Posts avec date seulement — Plan B."""
    return pd.DataFrame({
        "account_id": ["u3"] * 5 + ["u4"] * 5,
        "created_at": (
            pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03",
                            "2023-01-04", "2023-01-05"]).tolist()
            + pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01",
                              "2023-01-01", "2023-01-01"]).tolist()
        ),
    })


# ---------------------------------------------------------------------------
# Tests Plan A (timestamps précis)
# ---------------------------------------------------------------------------

class TestTemporalPlanA:

    def test_returns_dataframe(self, posts_precise):
        result = extract_temporal_features(posts_precise)
        assert isinstance(result, pd.DataFrame)

    def test_has_account_id_col(self, posts_precise):
        result = extract_temporal_features(posts_precise)
        assert "account_id" in result.columns

    def test_one_row_per_account(self, posts_precise):
        result = extract_temporal_features(posts_precise)
        n_accounts = posts_precise["account_id"].nunique()
        assert len(result) == n_accounts

    def test_plan_a_features_present(self, posts_precise):
        result = extract_temporal_features(posts_precise)
        # Colonnes avec préfixe t_ (ex: t_ipt_mean, t_night_ratio, t_hour_entropy)
        t_cols = [c for c in result.columns if c.startswith("t_")]
        assert len(t_cols) >= 3, f"Features temporelles Plan A attendues, trouvées : {t_cols}"

    def test_night_account_has_higher_night_ratio(self, posts_precise):
        """u2 poste la nuit (22h) → night_ratio doit être >= u1."""
        result = extract_temporal_features(posts_precise).set_index("account_id")
        night_cols = [c for c in result.columns if "night" in c.lower()]
        if night_cols:
            col = night_cols[0]
            assert result.loc["u2", col] >= result.loc["u1", col]


# ---------------------------------------------------------------------------
# Tests Plan B (dates seulement)
# ---------------------------------------------------------------------------

class TestTemporalPlanB:

    def test_plan_b_works_with_date_only(self, posts_date_only):
        result = extract_temporal_features(posts_date_only)
        assert not result.empty

    def test_dense_poster_detectable(self, posts_date_only):
        """u4 poste tout le même jour → moins de jours actifs que u3."""
        result = extract_temporal_features(posts_date_only).set_index("account_id")
        active_days_cols = [c for c in result.columns if "active_day" in c.lower()]
        if active_days_cols:
            col = active_days_cols[0]
            assert result.loc["u3", col] > result.loc["u4", col]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTemporalEdgeCases:

    def test_single_post_per_account(self):
        df = pd.DataFrame({
            "account_id": ["u1", "u2"],
            "created_at": ["2023-01-01T08:00:00Z", "2023-01-02T12:00:00Z"],
        })
        result = extract_temporal_features(df)
        assert len(result) == 2

    def test_no_timestamp_col_returns_dataframe(self):
        df = pd.DataFrame({"account_id": ["u1", "u2"], "text": ["a", "b"]})
        result = extract_temporal_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan_timestamps_no_crash(self):
        df = pd.DataFrame({
            "account_id": ["u1", "u1"],
            "created_at": [None, None],
        })
        result = extract_temporal_features(df)
        assert isinstance(result, pd.DataFrame)
