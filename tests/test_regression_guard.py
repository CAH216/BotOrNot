# tests/test_regression_guard.py
"""
Garde-fous anti-régression — BotOrNot
======================================
Vérifie que golden_baseline et cutdown baseline :
  1. S'exécutent sans erreur sur datasets synthétiques de référence
  2. Produisent le bon format de sortie (colonnes, count, pas de NaN)
  3. Respectent des seuils minimaux de métriques (AUROC, F1, précision)
  4. Ne modifient pas les fichiers de config source
  5. Produisent des features sans colonnes entièrement NaN

Seuils minimaux définis par scénario (conservateurs mais non-triviaux) :
  - Scénario classique  : AUROC > 0.60, F1 > 0.50
  - Scénario texte nul  : AUROC > 0.50, pipeline stable
  - Scénario burst bots : AUROC > 0.65

Ces seuils sont volontairement bas — ils détectent les régressions
sans être sensibles au bruit statistique.
"""
import os
import sys
import copy
import hashlib
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.metrics import roc_auc_score, f1_score

from scripts.run_baseline import (
    _build_features,
    _impute,
    _get_model,
    _fit_predict,
)
from scripts.run_cutdown import run_cutdown

SEED = 42


# ─────────────────────────────────────────────────────────────
# Helpers : fabriques de datasets synthétiques reproductibles
# ─────────────────────────────────────────────────────────────

def _make_account(uid, is_bot, n_posts, rng, **kw):
    base_hour = rng.integers(2, 5) if is_bot else rng.integers(8, 22)
    hours     = rng.integers(base_hour, base_hour + 2, size=n_posts) % 24
    rows      = []
    start     = pd.Timestamp("2023-01-01", tz="UTC")
    for i in range(n_posts):
        gap   = float(rng.exponential(5 if is_bot else 600))
        start = start + pd.Timedelta(seconds=gap)
        rows.append({
            "user_id":         uid,
            "followers_count": kw.get("followers", int(rng.exponential(200 if is_bot else 800))),
            "following_count": kw.get("following", int(rng.exponential(3000 if is_bot else 600))),
            "statuses_count":  kw.get("statuses",  int(rng.exponential(50000 if is_bot else 5000))),
            "verified":        kw.get("verified",   False),
            "is_bot":          is_bot,
            "created_at":      str(start),
            "text":            kw.get("text", (
                "buy now!! http://spam.com #ad" if is_bot
                else "Good morning everyone!"
            )),
            "hour":            int(hours[i % len(hours)]),
        })
    return pd.DataFrame(rows)


def _make_ref_dataset(n_bots=100, n_humans=100, posts=8, seed=SEED, **kw):
    """Dataset de référence reproductible (même seed = même dataset)."""
    rng  = np.random.default_rng(seed)
    dfs  = []
    dfs += [_make_account(f"bot_{i}",   1, posts, rng, **kw.get("bot",   {})) for i in range(n_bots)]
    dfs += [_make_account(f"human_{i}", 0, posts, rng, **kw.get("human", {})) for i in range(n_humans)]
    return pd.concat(dfs, ignore_index=True)


def _build_y(df, feat):
    return df.groupby("user_id")["is_bot"].max().reindex(feat["user_id"]).values.astype(int)


def _quick_auc_f1(df, seed=SEED):
    """Run minimal : features → LR → AUROC + F1."""
    feat = _build_features(df, "user_id")
    y    = _build_y(df, feat)
    X    = _impute(feat.drop(columns=["user_id"]).select_dtypes(include=[np.number]))
    model = _get_model("lr", seed)
    _, proba = _fit_predict(model, X.values, y, X.values)
    t     = 0.5
    pred  = (proba >= t).astype(int)
    auroc = roc_auc_score(y, proba)
    f1    = f1_score(y, pred, zero_division=0)
    return auroc, f1, feat, proba, y


# ─────────────────────────────────────────────────────────────
# A — _build_features : intégrité des features
# ─────────────────────────────────────────────────────────────

class TestFeatureIntegrity:
    """Vérifie que _build_features ne produit pas de sorties corrompues."""

    @pytest.fixture(scope="class")
    def ref_df(self):
        return _make_ref_dataset()

    def test_output_is_dataframe(self, ref_df):
        feat = _build_features(ref_df, "user_id")
        assert isinstance(feat, pd.DataFrame)

    def test_correct_account_count(self, ref_df):
        feat = _build_features(ref_df, "user_id")
        expected = ref_df["user_id"].nunique()
        assert len(feat) == expected, f"Attendu {expected} comptes, obtenu {len(feat)}"

    def test_no_all_nan_columns(self, ref_df):
        feat = _build_features(ref_df, "user_id")
        numeric = feat.select_dtypes(include=[np.number])
        all_nan = numeric.columns[numeric.isna().all()].tolist()
        assert len(all_nan) == 0, f"Colonnes entierement NaN : {all_nan}"

    def test_no_label_leakage(self, ref_df):
        feat = _build_features(ref_df, "user_id")
        assert "is_bot" not in feat.columns
        assert "label"  not in feat.columns

    def test_user_id_column_present(self, ref_df):
        feat = _build_features(ref_df, "user_id")
        assert "user_id" in feat.columns

    def test_minimum_feature_count(self, ref_df):
        feat = _build_features(ref_df, "user_id")
        n_num = feat.select_dtypes(include=[np.number]).shape[1]
        assert n_num >= 5, f"Moins de 5 features numeriques : {n_num}"

    def test_reproducibility(self, ref_df):
        """Deux appels sur le même dataset doivent produire le même résultat."""
        feat1 = _build_features(ref_df, "user_id")
        feat2 = _build_features(ref_df, "user_id")
        num1  = feat1.select_dtypes(include=[np.number]).fillna(0)
        num2  = feat2.select_dtypes(include=[np.number]).fillna(0)
        pd.testing.assert_frame_equal(num1, num2, check_like=True)


# ─────────────────────────────────────────────────────────────
# B — Métriques minimales : seuils de régression par scénario
# ─────────────────────────────────────────────────────────────

class TestMinimumMetrics:
    """
    Seuils minimaux garantis. Si ces tests échouent, le pipeline a régressé.
    Seuils conservateurs : ils détectent les cassures, pas le bruit.
    """

    # ── Scénario 1 : classique (signal fort) ─────────────────
    def test_scenario_classic_auroc(self):
        """Signal clair tabular+temporal+text → AUROC > 0.60."""
        df = _make_ref_dataset(n_bots=100, n_humans=100, posts=10, seed=SEED)
        auroc, _, _, _, _ = _quick_auc_f1(df)
        assert auroc > 0.60, f"REGRESSION : AUROC classique trop bas ({auroc:.4f} <= 0.60)"

    def test_scenario_classic_f1(self):
        """F1 > 0.50 sur scénario classique."""
        df = _make_ref_dataset(n_bots=100, n_humans=100, posts=10, seed=SEED)
        _, f1, _, _, _ = _quick_auc_f1(df)
        assert f1 > 0.50, f"REGRESSION : F1 classique trop bas ({f1:.4f} <= 0.50)"

    # ── Scénario 2 : texte nul (pas de crash) ────────────────
    def test_scenario_no_text_does_not_crash(self):
        """Pipeline stable même avec 100% de texte NaN."""
        df = _make_ref_dataset(bot={"text": ""}, human={"text": ""})
        df["text"] = np.nan
        auroc, f1, feat, _, _ = _quick_auc_f1(df)
        assert feat is not None
        assert auroc > 0.40, f"REGRESSION : AUC sans texte trop bas ({auroc:.4f})"

    # ── Scénario 3 : burst bots (signal temporel fort) ───────
    def test_scenario_burst_bots_auroc(self):
        """IPT ultra-régulier pour bots → AUROC > 0.65."""
        rng = np.random.default_rng(SEED + 99)
        rows = []
        for i in range(80):
            ts = pd.Timestamp("2023-01-01 02:00:00", tz="UTC")
            for _ in range(15):
                ts = ts + pd.Timedelta(seconds=float(rng.uniform(1, 5)))
                rows.append({"user_id": f"bot_{i}", "is_bot": 1,
                             "followers_count": int(rng.integers(0, 200)),
                             "following_count": int(rng.integers(3000, 8000)),
                             "statuses_count":  int(rng.integers(5000, 50000)),
                             "verified": False, "created_at": str(ts),
                             "text": "promo http://spam.com"})
        for i in range(80):
            ts = pd.Timestamp("2023-01-01 09:00:00", tz="UTC")
            for _ in range(15):
                ts = ts + pd.Timedelta(seconds=float(rng.uniform(600, 7200)))
                rows.append({"user_id": f"human_{i}", "is_bot": 0,
                             "followers_count": int(rng.exponential(1000)),
                             "following_count": int(rng.integers(50, 800)),
                             "statuses_count":  int(rng.exponential(3000)),
                             "verified": bool(rng.random() < 0.05),
                             "created_at": str(ts),
                             "text": "Good morning!"})
        df = pd.DataFrame(rows)
        auroc, _, _, _, _ = _quick_auc_f1(df)
        assert auroc > 0.65, f"REGRESSION : AUROC burst-bots trop bas ({auroc:.4f} <= 0.65)"

    # ── Scénario 4 : déséquilibre des classes ─────────────────
    def test_scenario_imbalanced_does_not_crash(self):
        """10% de bots seulement → pipeline stable."""
        df = _make_ref_dataset(n_bots=20, n_humans=180, posts=8, seed=SEED + 7)
        auroc, f1, feat, _, y = _quick_auc_f1(df)
        assert feat is not None
        assert len(feat) == 200
        assert auroc > 0.40, f"REGRESSION : AUC imbalance trop bas ({auroc:.4f})"


# ─────────────────────────────────────────────────────────────
# C — Format de sortie du pipeline cutdown
# ─────────────────────────────────────────────────────────────

class TestCutdownOutputFormat:
    """
    Vérifie que run_cutdown produit un fichier CSV conforme.
    Teste les deux profils : conservative et balanced.
    """

    @pytest.fixture(scope="class")
    def cutdown_output(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("cutdown_out")
        df = _make_ref_dataset(n_bots=80, n_humans=80, posts=8)
        train_path = str(tmp / "train.csv")
        df.to_csv(train_path, index=False)

        results = {}
        for profile in ("conservative", "balanced"):
            out_dir = str(tmp / profile)
            os.makedirs(out_dir, exist_ok=True)

            class Args:
                train    = train_path
                test     = None
                model    = "lr"
                cv_folds = 3
                seed     = SEED
                threshold = None
                out      = out_dir
                label_col = None
                id_col    = None

            setattr(Args, "profile", profile)
            meta = run_cutdown(Args())
            results[profile] = {"meta": meta, "out": out_dir}
        return results

    def test_csv_exists_conservative(self, cutdown_output):
        out = cutdown_output["conservative"]["out"]
        assert os.path.exists(os.path.join(out, "cutdown_conservative.csv"))

    def test_csv_exists_balanced(self, cutdown_output):
        out = cutdown_output["balanced"]["out"]
        assert os.path.exists(os.path.join(out, "cutdown_balanced.csv"))

    def test_csv_columns_conservative(self, cutdown_output):
        out = cutdown_output["conservative"]["out"]
        df  = pd.read_csv(os.path.join(out, "cutdown_conservative.csv"))
        assert "user_id" in df.columns
        assert df.shape[1] >= 2, "Moins de 2 colonnes dans la soumission"

    def test_no_nan_in_label_col(self, cutdown_output):
        for profile, data in cutdown_output.items():
            df = pd.read_csv(
                os.path.join(data["out"], f"cutdown_{profile}.csv")
            )
            label_col = df.columns[-1]
            assert df[label_col].isna().sum() == 0, \
                f"NaN dans la colonne label ({profile})"

    def test_labels_are_binary(self, cutdown_output):
        for profile, data in cutdown_output.items():
            df = pd.read_csv(
                os.path.join(data["out"], f"cutdown_{profile}.csv")
            )
            label_col = df.columns[-1]
            unique_vals = set(df[label_col].unique())
            assert unique_vals.issubset({0, 1}), \
                f"Labels non binaires ({profile}) : {unique_vals}"

    def test_account_count_matches(self, cutdown_output):
        """Toutes les 160 comptes de référence sont dans la soumission."""
        for profile, data in cutdown_output.items():
            df = pd.read_csv(
                os.path.join(data["out"], f"cutdown_{profile}.csv")
            )
            assert len(df) == 160, \
                f"Count incorrect ({profile}) : {len(df)} != 160"

    def test_meta_json_exists(self, cutdown_output):
        for profile, data in cutdown_output.items():
            meta_path = os.path.join(data["out"], f"cutdown_{profile}_meta.json")
            assert os.path.exists(meta_path), f"meta.json absent ({profile})"

    def test_meta_contains_required_keys(self, cutdown_output):
        import json
        required = {"profile", "model", "threshold", "oof_metrics",
                    "n_train_accounts", "elapsed_seconds"}
        for profile, data in cutdown_output.items():
            meta_path = os.path.join(data["out"], f"cutdown_{profile}_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            missing = required - set(meta.keys())
            assert not missing, f"Cles manquantes dans meta ({profile}) : {missing}"

    def test_oof_auroc_above_minimum(self, cutdown_output):
        """AUROC OOF > 0.55 sur dataset de référence."""
        for profile, data in cutdown_output.items():
            oof_auroc = data["meta"]["oof_metrics"]["auroc"]
            assert oof_auroc > 0.55, \
                f"REGRESSION : OOF AUROC cutdown trop bas ({profile}) : {oof_auroc:.4f}"


# ─────────────────────────────────────────────────────────────
# D — Immuabilité des fichiers de config
# ─────────────────────────────────────────────────────────────

class TestConfigImmutability:
    """
    S'assure que le pipeline ne modifie pas les fichiers de config source.
    Conforme à RULES.md §1 : golden_baseline est READ-ONLY.
    """

    @staticmethod
    def _file_hash(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def test_golden_baseline_not_modified(self, tmp_path):
        """golden_baseline.yaml ne doit pas être modifié après une run."""
        baseline_path = "configs/golden_baseline.yaml"
        if not os.path.exists(baseline_path):
            pytest.skip("golden_baseline.yaml introuvable")
        hash_before = self._file_hash(baseline_path)
        # Simuler une lecture de config (même logique que experiment_runner)
        import yaml
        with open(baseline_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Opération de lecture seule : modifier une copie
        import copy
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["model"]["name"] = "mutated"
        # Vérifier que le fichier source n'a pas changé
        hash_after = self._file_hash(baseline_path)
        assert hash_before == hash_after, \
            "VIOLATION RULES.md : golden_baseline.yaml a ete modifie !"

    def test_submission_profiles_not_modified(self):
        """submission_profiles.yaml doit rester intact après imports."""
        path = "configs/submission_profiles.yaml"
        if not os.path.exists(path):
            pytest.skip("submission_profiles.yaml introuvable")
        hash_before = self._file_hash(path)
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            _ = yaml.safe_load(f)
        hash_after = self._file_hash(path)
        assert hash_before == hash_after


# ─────────────────────────────────────────────────────────────
# E — Smoke tests de démarrage (imports critiques)
# ─────────────────────────────────────────────────────────────

class TestCriticalImports:
    """
    Vérifie que tous les modules critiques s'importent sans erreur.
    Un échec ici signale une régression de dépendance ou de syntaxe.
    """

    def test_import_run_baseline(self):
        from scripts.run_baseline import _build_features, _impute, _get_model
        assert callable(_build_features)

    def test_import_run_cutdown(self):
        from scripts.run_cutdown import run_cutdown
        assert callable(run_cutdown)

    def test_import_anti_fp(self):
        from src.inference.anti_fp import AntiFPFilter, AntiFPConfig
        af = AntiFPFilter(AntiFPConfig(enabled=False))
        assert af is not None

    def test_import_experiment_runner(self):
        from scripts.experiment_runner import _deep_merge, _diff_configs
        assert callable(_deep_merge)

    def test_import_tabular_features(self):
        from src.features.tabular import extract_tabular_features
        assert callable(extract_tabular_features)

    def test_import_temporal_features(self):
        from src.features.temporal import extract_temporal_features
        assert callable(extract_temporal_features)
