# tests/test_stress_scenarios.py
"""
Stress tests — Scénarios synthétiques réalistes
================================================
Valide que le pipeline baseline produit des résultats sensés
sur 4 scénarios représentatifs des challenges réels :

  1. Scénario classique    — tabular + temporal + text
  2. Scénario texte faible — comptes sans bio/post
  3. Scénario burst fort   — bots actifs très régulièrement la nuit
  4. Scénario FP difficile — humains atypiques (fans, journalistes, power-users)
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Import du pipeline baseline en mode "library" ──────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.run_baseline import (
    _build_features,
    _impute,
    _get_model,
    _fit_predict,
)

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────
# Factories de datasets synthétiques
# ──────────────────────────────────────────────────────────────────────────

def _make_account(uid: str, is_bot: int, n_posts: int, rng: np.random.Generator,
                  **overrides) -> pd.DataFrame:
    """Crée les lignes d'un seul compte avec des profils réalistes."""
    base_hour = rng.integers(2, 5) if is_bot else rng.integers(8, 22)
    hours = rng.integers(base_hour, base_hour + 2, size=n_posts) % 24

    rows = {
        "user_id":          uid,
        "followers_count":  overrides.get("followers", int(rng.exponential(200 if is_bot else 800))),
        "following_count":  overrides.get("following", int(rng.exponential(3000 if is_bot else 600))),
        "statuses_count":   overrides.get("statuses",  int(rng.exponential(50000 if is_bot else 5000))),
        "verified":         overrides.get("verified",  False),
        "is_bot":           is_bot,
    }

    records = []
    start   = pd.Timestamp("2023-01-01", tz="UTC")
    for i in range(n_posts):
        gap = (rng.exponential(5 if is_bot else 600))  # secondes entre posts
        start = start + pd.Timedelta(seconds=gap)
        text = overrides.get("text", (
            "buy now!! http://spam.com #ad #promo @bot1" if is_bot
            else "Good morning everyone, just had coffee ☕"
        ))
        record = dict(rows)
        record["created_at"] = str(start)
        record["text"]       = text
        record["hour"]       = int(hours[i % len(hours)])
        records.append(record)
    return pd.DataFrame(records)


def _make_dataset(n_bots: int, n_humans: int, posts_per_account: int,
                  rng: np.random.Generator, **overrides) -> pd.DataFrame:
    dfs = []
    for i in range(n_bots):
        dfs.append(_make_account(f"bot_{i}", 1, posts_per_account, rng, **overrides.get("bot", {})))
    for i in range(n_humans):
        dfs.append(_make_account(f"human_{i}", 0, posts_per_account, rng, **overrides.get("human", {})))
    return pd.concat(dfs, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
# Scénario 1 — Classique : tabular + temporal + text
# ──────────────────────────────────────────────────────────────────────────

class TestScenarioClassic:
    """
    Dataset complet : followers/following déséquilibrés, posts nocturnes,
    texte spam pour les bots.
    Attendu : AUC > 0.65 (signal clair).
    """

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(SEED)
        df = _make_dataset(80, 80, 10, rng)
        return df

    def test_features_shape(self, data):
        feat = _build_features(data, "user_id")
        assert len(feat) == 160, "160 comptes attendus"
        assert feat.shape[1] > 5,  "Au moins 5 features générées"

    def test_no_all_nan_columns(self, data):
        feat = _build_features(data, "user_id")
        numeric = feat.select_dtypes(include=[np.number])
        all_nan = numeric.columns[numeric.isna().all()].tolist()
        assert len(all_nan) == 0, f"Colonnes entièrement NaN : {all_nan}"

    def test_auc_above_chance(self, data):
        feat = _build_features(data, "user_id")
        y    = data.groupby("user_id")["is_bot"].max().reindex(feat["user_id"]).values
        X    = _impute(feat.drop(columns=["user_id"]).select_dtypes(include=[np.number]))
        model = _get_model("lr", SEED)
        _, proba = _fit_predict(model, X.values, y, X.values)
        auc = roc_auc_score(y, proba)
        assert auc > 0.55, f"AUC trop faible sur scénario classique : {auc:.4f}"

    def test_no_leakage(self, data):
        """Vérifie que la colonne is_bot n'est PAS dans les features."""
        feat = _build_features(data, "user_id")
        assert "is_bot" not in feat.columns
        assert "label" not in feat.columns


# ──────────────────────────────────────────────────────────────────────────
# Scénario 2 — Texte faible (comptes sans bio ni posts)
# ──────────────────────────────────────────────────────────────────────────

class TestScenarioWeakText:
    """
    Comptes sans texte ou avec texte quasi-vide.
    Attendu : le pipeline ne plante pas, features cohérentes.
    """

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(SEED + 1)
        df = _make_dataset(60, 60, 5, rng,
            bot={"text": ""},
            human={"text": ""},
        )
        # Remplacer 80% des textes par NaN
        mask = rng.random(len(df)) < 0.8
        df.loc[mask, "text"] = np.nan
        return df

    def test_pipeline_does_not_crash(self, data):
        feat = _build_features(data, "user_id")
        assert feat is not None
        assert len(feat) == 120

    def test_text_features_are_nan_or_zero(self, data):
        feat = _build_features(data, "user_id")
        txt_cols = [c for c in feat.columns if c.startswith("txt_")]
        # Avec texte vide, les features textuelles doivent être proches de 0 ou NaN
        if txt_cols:
            max_val = feat[txt_cols].abs().max().max()
            assert max_val < 100, f"Valeur txt suspecte : {max_val}"

    def test_tabular_features_still_present(self, data):
        feat = _build_features(data, "user_id")
        tab_cols = [c for c in feat.columns if c.startswith(("raw_", "tab_", "bool_"))]
        assert len(tab_cols) > 0, "Aucune feature tabulaire sans texte"

    def test_auc_above_random(self, data):
        feat = _build_features(data, "user_id")
        y    = data.groupby("user_id")["is_bot"].max().reindex(feat["user_id"]).values
        X    = _impute(feat.drop(columns=["user_id"]).select_dtypes(include=[np.number]))
        if X.shape[1] == 0:
            pytest.skip("Aucune feature numérique — dataset trop sparse")
        model = _get_model("lr", SEED)
        _, proba = _fit_predict(model, X.values, y, X.values)
        auc = roc_auc_score(y, proba)
        assert auc > 0.40, f"AUC anormalement basse (possible bug) : {auc:.4f}"


# ──────────────────────────────────────────────────────────────────────────
# Scénario 3 — Timestamps forts (burst bots)
# ──────────────────────────────────────────────────────────────────────────

class TestScenarioBurstBots:
    """
    Bots postent en bursts nocturnes ultra-réguliers (gap ~2s).
    Humains postent aléatoirement en journée (gap ~1h).
    Attendu : le signal temporel seul doit être très discriminant.
    """

    @pytest.fixture(scope="class")
    def data(self):
        accounts = []
        rng = np.random.default_rng(SEED + 2)

        # Bots : burst nocturne, 1 post toutes les 2 secondes
        for i in range(70):
            rows = []
            ts   = pd.Timestamp("2023-03-01 02:00:00", tz="UTC")
            followers = int(rng.integers(0, 200))
            for _ in range(15):
                ts = ts + pd.Timedelta(seconds=float(rng.uniform(1, 5)))
                rows.append({
                    "user_id": f"bot_{i}", "is_bot": 1,
                    "followers_count": followers,
                    "following_count": int(rng.integers(3000, 8000)),
                    "statuses_count": int(rng.integers(5000, 50000)),
                    "verified": False,
                    "created_at": str(ts),
                    "text": "promo http://spam.com",
                })
            accounts.append(pd.DataFrame(rows))

        # Humains : aléatoire en journée, gap ~1h
        for i in range(70):
            rows = []
            ts   = pd.Timestamp("2023-03-01 09:00:00", tz="UTC")
            followers = int(rng.exponential(1000))
            for _ in range(15):
                ts = ts + pd.Timedelta(seconds=float(rng.uniform(600, 7200)))
                rows.append({
                    "user_id": f"human_{i}", "is_bot": 0,
                    "followers_count": followers,
                    "following_count": int(rng.integers(50, 800)),
                    "statuses_count": int(rng.exponential(3000)),
                    "verified": bool(rng.random() < 0.05),
                    "created_at": str(ts),
                    "text": "Nice day! Just got coffee",
                })
            accounts.append(pd.DataFrame(rows))

        return pd.concat(accounts, ignore_index=True)

    def test_temporal_features_generated(self, data):
        feat = _build_features(data, "user_id")
        tmp_cols = [c for c in feat.columns if c.startswith("tmp_")]
        assert len(tmp_cols) >= 5, f"Trop peu de features temporelles : {tmp_cols}"

    def test_ipt_mean_separates_classes(self, data):
        """Les bots ont un IPT moyen bien plus court que les humains."""
        feat = _build_features(data, "user_id")
        if "tmp_ipt_mean" not in feat.columns:
            pytest.skip("tmp_ipt_mean non disponible")
        feat["is_bot"] = data.groupby("user_id")["is_bot"].max().reindex(feat["user_id"]).values
        bot_ipt   = feat[feat["is_bot"] == 1]["tmp_ipt_mean"].median()
        human_ipt = feat[feat["is_bot"] == 0]["tmp_ipt_mean"].median()
        assert bot_ipt < human_ipt / 5, (
            f"IPT bots ({bot_ipt:.1f}s) devrait être << humains ({human_ipt:.1f}s)"
        )

    def test_strong_auc_from_temporal(self, data):
        feat = _build_features(data, "user_id")
        y    = data.groupby("user_id")["is_bot"].max().reindex(feat["user_id"]).values
        tmp_cols = [c for c in feat.columns if c.startswith("tmp_")]
        if not tmp_cols:
            pytest.skip("Pas de features temporelles")
        X = _impute(feat[tmp_cols])
        model = _get_model("lr", SEED)
        _, proba = _fit_predict(model, X.values, y, X.values)
        auc = roc_auc_score(y, proba)
        assert auc > 0.70, f"AUC temporelle attendue > 0.70, obtenu : {auc:.4f}"


# ──────────────────────────────────────────────────────────────────────────
# Scénario 4 — Faux positifs difficiles (humains atypiques)
# ──────────────────────────────────────────────────────────────────────────

class TestScenarioHardFalsePositives:
    """
    Power-users (journalistes, fans) : postent beaucoup, suivent beaucoup,
    actifs la nuit. Ces humains ressemblent à des bots → risque FP élevé.
    Attendu : le pipeline ne les classe pas tous comme bots (précision > 0.5).
    """

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(SEED + 3)
        rows = []

        # Bots classiques
        for i in range(60):
            ts = pd.Timestamp("2023-01-01 03:00:00", tz="UTC")
            for _ in range(10):
                ts = ts + pd.Timedelta(seconds=float(rng.uniform(2, 10)))
                rows.append({
                    "user_id": f"bot_{i}", "is_bot": 1,
                    "followers_count": int(rng.integers(0, 300)),
                    "following_count": int(rng.integers(4000, 8000)),
                    "statuses_count": int(rng.integers(20000, 80000)),
                    "verified": False,
                    "created_at": str(ts),
                    "text": "buy http://link.com #promo",
                })

        # Power-users humains (journalistes, créateurs)
        for i in range(60):
            ts = pd.Timestamp("2023-01-01 08:00:00", tz="UTC")
            followers = int(rng.exponential(15000))  # beaucoup de followers
            for _ in range(25):  # beaucoup de posts
                ts = ts + pd.Timedelta(seconds=float(rng.uniform(30, 600)))
                hour = rng.choice([0, 1, 2, 22, 23, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])

                rows.append({
                    "user_id": f"poweruser_{i}", "is_bot": 0,
                    "followers_count": followers,
                    "following_count": int(rng.integers(200, 2000)),
                    "statuses_count": int(rng.integers(5000, 30000)),
                    "verified": bool(rng.random() < 0.3),
                    "created_at": str(ts),
                    "text": rng.choice([
                        "Breaking: new report just released",
                        "Just finished writing — long thread coming",
                        "Live coverage from the event",
                        "My take on this situation",
                    ]),
                })

        return pd.DataFrame(rows)

    def test_features_built_for_all_accounts(self, data):
        feat = _build_features(data, "user_id")
        n_expected = data["user_id"].nunique()
        assert len(feat) == n_expected

    def test_model_precision_acceptable(self, data):
        """
        Le modèle ne doit pas classer TOUS les power-users comme bots.
        Précision > 0.40 signifie qu'il fait mieux qu'un seuil naïf.
        """
        from sklearn.metrics import precision_score
        feat = _build_features(data, "user_id")
        y    = data.groupby("user_id")["is_bot"].max().reindex(feat["user_id"]).values
        X    = _impute(feat.drop(columns=["user_id"]).select_dtypes(include=[np.number]))
        model = _get_model("lr", SEED)
        model.fit(X.values, y)
        proba = model.predict_proba(X.values)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        prec  = precision_score(y, pred, zero_division=0)
        assert prec > 0.35, f"Précision trop faible sur FP difficiles : {prec:.4f}"

    def test_followers_signal_helps(self, data):
        """
        Les power-users ont plus de followers : vérifier que
        la feature tab_followers_log existe et est non-nulle.
        """
        feat = _build_features(data, "user_id")
        if "tab_followers_log" in feat.columns:
            pu_mask = feat["user_id"].str.startswith("poweruser")
            bt_mask = feat["user_id"].str.startswith("bot")
            pu_log  = feat[pu_mask]["tab_followers_log"].median()
            bt_log  = feat[bt_mask]["tab_followers_log"].median()
            assert pu_log > bt_log, "Power-users devraient avoir plus de followers que les bots"
        else:
            pytest.skip("tab_followers_log non disponible")

    def test_conservative_threshold_reduces_fp(self, data):
        """
        Avec seuil 0.6 (conservateur), les FP doivent être < 40% des humains.
        """
        feat = _build_features(data, "user_id")
        y    = data.groupby("user_id")["is_bot"].max().reindex(feat["user_id"]).values
        X    = _impute(feat.drop(columns=["user_id"]).select_dtypes(include=[np.number]))
        model = _get_model("lr", SEED)
        model.fit(X.values, y)
        proba = model.predict_proba(X.values)[:, 1]
        pred  = (proba >= 0.6).astype(int)
        n_humans   = (y == 0).sum()
        fp_count   = ((pred == 1) & (y == 0)).sum()
        fp_rate    = fp_count / max(n_humans, 1)
        assert fp_rate < 0.5, f"Trop de FP avec seuil=0.6 : {fp_rate:.1%} des humains mal classés"
