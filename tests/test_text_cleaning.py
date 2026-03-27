# tests/test_text_cleaning.py
"""Tests unitaires — nettoyage de texte et parsing de dates."""
import pytest
import pandas as pd
import numpy as np

from src.preprocessing.clean_text import clean_text, clean_posts_df
from src.preprocessing.parse_dates import preprocess_dates, parse_date_column


# Helpers
def _df_from_result(result):
    """Déroule (df, extra) si tuple."""
    if isinstance(result, tuple):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Tests clean_text (niveau chaîne)
# ---------------------------------------------------------------------------

class TestCleanText:

    def test_returns_string(self):
        result = clean_text("Hello world!")
        assert isinstance(result, str)

    def test_none_handled(self):
        result = clean_text(None)
        assert isinstance(result, str)

    def test_url_removal(self):
        result = clean_text("Check this http://example.com out", remove_urls=True)
        assert "http" not in result

    def test_url_kept_when_not_removing(self):
        result = clean_text("Visit http://example.com", remove_urls=False)
        assert "http" in result

    def test_whitespace_normalized(self):
        result = clean_text("hello   world\t\nfoo")
        assert "  " not in result

    def test_empty_string(self):
        result = clean_text("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests clean_posts_df (niveau DataFrame)
# ---------------------------------------------------------------------------

class TestCleanPostsDf:

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "account_id": ["u1", "u2"],
            "text": ["Hello http://spam.com", "Good morning @user"],
        })
        result = clean_posts_df(df)
        assert isinstance(result, pd.DataFrame)

    def test_same_length(self):
        df = pd.DataFrame({
            "account_id": ["u1", "u2", "u3"],
            "text": ["a", "b", None],
        })
        result = clean_posts_df(df)
        assert len(result) == 3

    def test_text_col_present_in_output(self):
        df = pd.DataFrame({"account_id": ["u1"], "text": ["Hello world"]})
        result = clean_posts_df(df)
        has_text = "text_clean" in result.columns or "text" in result.columns
        assert has_text

    def test_no_crash_on_all_nulls(self):
        df = pd.DataFrame({"account_id": ["u1", "u2"], "text": [None, None]})
        result = clean_posts_df(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests parse_date_column
# ---------------------------------------------------------------------------

class TestParseDateColumn:

    def test_iso_timestamps_parsed(self):
        df = pd.DataFrame({"created_at": ["2023-01-01T08:00:00Z", "2023-06-15T12:30:00Z"]})
        result = parse_date_column(df, "created_at")
        # Retourne DataFrame ou Series avec datetime
        if isinstance(result, pd.DataFrame):
            col = result["created_at"] if "created_at" in result.columns else result.iloc[:, 0]
        else:
            col = result
        assert pd.api.types.is_datetime64_any_dtype(col)

    def test_handles_none(self):
        df = pd.DataFrame({"created_at": [None, "2023-01-01T08:00:00Z", np.nan]})
        result = parse_date_column(df, "created_at")
        assert result is not None
        result_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame({"created_at": result})
        assert len(result_df) == 3


# ---------------------------------------------------------------------------
# Tests preprocess_dates (peut retourner tuple)
# ---------------------------------------------------------------------------

class TestPreprocessDates:

    def test_returns_something(self):
        df = pd.DataFrame({"created_at": ["2023-01-01T08:00:00Z"]})
        result = preprocess_dates(df)
        assert result is not None

    def test_extracts_hour(self):
        df = pd.DataFrame({"created_at": ["2023-01-01T08:00:00Z", "2023-01-01T20:00:00Z"]})
        result_df = _df_from_result(preprocess_dates(df))
        hour_cols = [c for c in result_df.columns if "hour" in c.lower()]
        assert len(hour_cols) >= 1

    def test_extracts_weekday(self):
        df = pd.DataFrame({"created_at": ["2023-01-01T08:00:00Z"]})
        result_df = _df_from_result(preprocess_dates(df))
        wd_cols = [c for c in result_df.columns
                   if "weekday" in c.lower() or "dayofweek" in c.lower()]
        assert len(wd_cols) >= 1

    def test_no_date_col_returns_unchanged(self):
        df = pd.DataFrame({"text": ["hello", "world"]})
        result_df = _df_from_result(preprocess_dates(df))
        assert isinstance(result_df, pd.DataFrame)
        assert "text" in result_df.columns

    def test_handles_null_dates(self):
        df = pd.DataFrame({"created_at": [None, "2023-01-01T08:00:00Z"]})
        result_df = _df_from_result(preprocess_dates(df))
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
