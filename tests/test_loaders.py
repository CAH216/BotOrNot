# tests/test_loaders.py
"""Tests unitaires — chargement et normalisation des données."""
import pytest
import pandas as pd

from src.data.loaders import load_bundle
from src.preprocessing.normalize_columns import normalize_columns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CSV_CONTENT = """user_id,text,created_at,label
u1,Hello world,2023-01-01T08:00:00Z,0
u2,Buy now click,2023-01-01T09:00:00Z,1
u3,Good morning,2023-01-01T10:00:00Z,0
"""

JSON_CONTENT = """[
  {"user_id": "u1", "text": "Hello", "created_at": "2023-01-01T08:00:00Z", "label": 0},
  {"user_id": "u2", "text": "Spam", "created_at": "2023-01-01T09:00:00Z", "label": 1}
]"""


@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(CSV_CONTENT, encoding="utf-8")
    return str(f)


@pytest.fixture
def json_file(tmp_path):
    f = tmp_path / "data.json"
    f.write_text(JSON_CONTENT, encoding="utf-8")
    return str(f)


# Helpers
def _normalize(df):
    """Déroule le tuple (df, report) retourné par normalize_columns."""
    result = normalize_columns(df)
    if isinstance(result, tuple):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Tests normalize_columns
# ---------------------------------------------------------------------------

class TestNormalizeColumns:

    def test_renames_user_id_to_account_id(self):
        df = pd.DataFrame({"user_id": ["u1", "u2"], "text": ["a", "b"]})
        result = _normalize(df)
        assert "account_id" in result.columns
        assert "user_id" not in result.columns

    def test_renames_content_to_text(self):
        df = pd.DataFrame({"account_id": ["u1"], "content": ["hello"]})
        result = _normalize(df)
        assert "text" in result.columns

    def test_renames_timestamp_variants(self):
        """timestamp et post_time doivent être renommés en created_at."""
        for raw_col, expected in [("timestamp", "created_at"), ("post_time", "created_at")]:
            df = pd.DataFrame({"account_id": ["u1"], raw_col: ["2023-01-01"]})
            result = _normalize(df)
            # La colonne doit exister sous son nom canonique
            assert raw_col not in result.columns or expected in result.columns, \
                f"Colonne '{raw_col}' non renommée"

    def test_unknown_columns_kept(self):
        df = pd.DataFrame({"account_id": ["u1"], "my_custom_col": [42]})
        result = _normalize(df)
        assert "my_custom_col" in result.columns

    def test_returns_dataframe(self):
        df = pd.DataFrame({"user_id": ["u1"], "text": ["hi"]})
        result = _normalize(df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe_handled(self):
        df = pd.DataFrame()
        # Ne doit pas planter
        result = normalize_columns(df)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests load_bundle
# ---------------------------------------------------------------------------

class TestLoadBundle:

    def test_load_csv(self, csv_file):
        bundle = load_bundle(csv_file)
        assert bundle is not None

    def test_load_json(self, json_file):
        bundle = load_bundle(json_file)
        assert bundle is not None

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_bundle("/chemin/qui/nexiste/pas.csv")
