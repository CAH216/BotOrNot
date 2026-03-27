#!/usr/bin/env python
"""
inspect_dataset.py — Script d'inspection ultra-rapide (~30 secondes)
=====================================================================
Usage:
    python scripts/inspect_dataset.py data/train.csv
    python scripts/inspect_dataset.py data/train.csv --edges data/edges.csv
    python scripts/inspect_dataset.py data/train.json --output report.json
"""
import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("❌ pandas et numpy requis : pip install pandas numpy")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — colonnes canoniques et patterns à rechercher
# ─────────────────────────────────────────────────────────────────────────────

# Patterns de noms pour chaque signal
PATTERNS = {
    "account_id": ["user_id", "account_id", "author_id", "userid", "uid", "id"],
    "text":       ["text", "content", "tweet", "post", "body", "message", "description", "bio"],
    "timestamp":  ["created_at", "timestamp", "date", "time", "posted_at", "post_time", "datetime"],
    "label":      ["label", "is_bot", "bot", "class", "target", "y", "genuine", "fake"],
    "followers":  ["followers", "followers_count", "follower_count", "num_followers"],
    "following":  ["following", "following_count", "friends_count", "num_following"],
    "screen_name":["screen_name", "username", "name", "handle"],
    "statuses":   ["statuses_count", "status_count", "tweet_count", "post_count", "num_posts"],
    "verified":   ["verified", "is_verified", "account_verified"],
    "default_pfp":["default_profile_image", "default_pfp", "has_avatar"],
    "retweet":    ["retweet_count", "rt_count", "retweets"],
    "source":     ["source", "client", "via", "app"],
    "lang":       ["lang", "language", "locale"],
}


def _match_cols(df_cols, patterns):
    """Retourne {signal: [col exacte]} pour les colonnes correspondantes."""
    result = {}
    lower_map = {c.lower().replace(" ", "_"): c for c in df_cols}
    for signal, pats in patterns.items():
        found = [lower_map[p] for p in pats if p in lower_map]
        if found:
            result[signal] = found
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        sys.exit(f"❌ Fichier introuvable : {path}")
    if ext == ".csv":
        df = pd.read_csv(path, nrows=100_000, low_memory=False)
    elif ext in (".json", ".jsonl"):
        try:
            df = pd.read_json(path, lines=(ext == ".jsonl"))
        except Exception:
            df = pd.read_json(path)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t", nrows=100_000, low_memory=False)
    else:
        sys.exit(f"❌ Format non supporté : {ext}. Attendu : csv, json, jsonl, tsv, parquet.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Analyse des colonnes
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_columns(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    missing = (df.isnull().sum() / n_rows * 100).round(1)

    col_details = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        miss_pct = missing[col]
        n_unique = df[col].nunique(dropna=True)
        sample = df[col].dropna().head(3).tolist()
        # Truncate long strings in sample
        sample = [str(v)[:50] + "…" if len(str(v)) > 50 else str(v) for v in sample]
        col_details.append({
            "column":     col,
            "dtype":      dtype,
            "missing_%":  miss_pct,
            "n_unique":   int(n_unique),
            "sample":     sample,
        })
    return col_details


# ─────────────────────────────────────────────────────────────────────────────
# Détection des signaux
# ─────────────────────────────────────────────────────────────────────────────

def _detect_signals(df: pd.DataFrame, matches: dict) -> dict:
    signals = {}

    # --- Texte ---
    has_text = "text" in matches
    if has_text:
        text_col = matches["text"][0]
        non_null = df[text_col].dropna()
        avg_len  = non_null.astype(str).str.len().mean()
        signals["text"] = {
            "found":   True,
            "column":  text_col,
            "avg_length": round(avg_len, 1),
            "empty_pct": round((df[text_col].isna().sum() +
                                (df[text_col] == "").sum()) / len(df) * 100, 1),
        }
    else:
        signals["text"] = {"found": False}

    # --- Timestamps ---
    has_ts = "timestamp" in matches
    if has_ts:
        ts_col = matches["timestamp"][0]
        try:
            ts_parsed = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            valid = ts_parsed.notna()
            signals["timestamps"] = {
                "found":       True,
                "column":      ts_col,
                "valid_pct":   round(valid.mean() * 100, 1),
                "min":         str(ts_parsed[valid].min()),
                "max":         str(ts_parsed[valid].max()),
                "granularity": _guess_granularity(ts_parsed[valid]),
            }
        except Exception as e:
            signals["timestamps"] = {"found": True, "column": ts_col, "error": str(e)}
    else:
        signals["timestamps"] = {"found": False}

    # --- Labels ---
    has_label = "label" in matches
    if has_label:
        label_col = matches["label"][0]
        vc = df[label_col].value_counts(dropna=True)
        signals["label"] = {
            "found":       True,
            "column":      label_col,
            "distribution": {str(k): int(v) for k, v in vc.items()},
            "n_classes":   int(vc.nunique()),
            "balance":     _balance_quality(vc),
        }
    else:
        signals["label"] = {"found": False}

    # --- Compte/nœud ---
    has_account = "account_id" in matches
    signals["account_id"] = {
        "found":     has_account,
        "column":    matches["account_id"][0] if has_account else None,
        "n_unique":  int(df[matches["account_id"][0]].nunique()) if has_account else None,
    }

    # --- Méta-profil ---
    profile_signals = {k: v[0] for k, v in matches.items()
                       if k in ("followers", "following", "statuses", "verified",
                                "screen_name", "default_pfp", "source")}
    signals["profile_features"] = {
        "found": len(profile_signals) > 0,
        "columns": profile_signals,
    }

    return signals


def _guess_granularity(series: pd.Series) -> str:
    """Devine la granularité temporelle."""
    if series.empty:
        return "unknown"
    try:
        hours = series.dt.hour
        minutes = series.dt.minute
        seconds = series.dt.second
        if seconds.std() > 0:
            return "second"
        if minutes.std() > 0:
            return "minute"
        if hours.std() > 0:
            return "hour"
        return "day"
    except Exception:
        return "unknown"


def _balance_quality(vc: pd.Series) -> str:
    if len(vc) < 2:
        return "single_class"
    ratio = vc.min() / vc.max()
    if ratio > 0.7:
        return "balanced"
    if ratio > 0.3:
        return "moderate_imbalance"
    return "high_imbalance"


# ─────────────────────────────────────────────────────────────────────────────
# Détection unité de prédiction
# ─────────────────────────────────────────────────────────────────────────────

def _detect_prediction_unit(df: pd.DataFrame, matches: dict, col_details: list) -> str:
    """Devine si on prédit au niveau compte ou au niveau post."""
    has_account = "account_id" in matches
    has_text    = "text" in matches

    if has_account and has_text:
        id_col = matches["account_id"][0]
        n_rows_per_account = len(df) / df[id_col].nunique()
        if n_rows_per_account > 1.5:
            return "post (plusieurs lignes par compte)"
        return "compte (une ligne par compte)"
    if has_account and not has_text:
        return "compte (profil uniquement)"
    if has_text and not has_account:
        return "post (pas d'ID de compte détecté)"
    return "indéterminé"


# ─────────────────────────────────────────────────────────────────────────────
# Recommandations de modules
# ─────────────────────────────────────────────────────────────────────────────

def _recommend_modules(signals: dict, has_edges: bool, n_rows: int) -> dict:
    recs = {}

    # Tabular (toujours)
    recs["tabular"] = {
        "activate": True,
        "reason": "Module baseline — toujours activé.",
    }

    # Text features
    if signals["text"]["found"]:
        avg_len = signals["text"].get("avg_length", 0)
        recs["text_basic"] = {
            "activate": True,
            "reason": f"Texte détecté (longueur moy. {avg_len} chars). TF-IDF léger activé.",
        }
        if avg_len > 30 and n_rows <= 50_000:
            recs["text_embeddings"] = {
                "activate": True,
                "reason": "Texte suffisamment long + dataset modéré → sentence-transformers possible.",
            }
        else:
            recs["text_embeddings"] = {
                "activate": False,
                "reason": "Texte court ou dataset trop large → embeddings non recommandés.",
            }
        recs["text_model"] = {
            "activate": True,
            "reason": "Modèle texte indépendant (TF-IDF) pour score texte fusionnable.",
        }
    else:
        for m in ("text_basic", "text_embeddings", "text_model"):
            recs[m] = {"activate": False, "reason": "Aucune colonne texte détectée."}

    # Temporal
    if signals["timestamps"]["found"]:
        recs["temporal"] = {
            "activate": True,
            "reason": f"Timestamps détectés (granularité: {signals['timestamps'].get('granularity','?')}).",
        }
    else:
        recs["temporal"] = {
            "activate": False,
            "reason": "Pas de colonne timestamp → module temporel désactivé.",
        }

    # Structural
    has_profile = signals["profile_features"]["found"]
    recs["structural"] = {
        "activate": has_profile,
        "reason": (
            f"Signaux structurels trouvés : {list(signals['profile_features']['columns'].keys())}."
            if has_profile else "Pas de colonnes de profil (followers, source…)."
        ),
    }

    # Relational
    recs["relational"] = {
        "activate": has_edges,
        "reason": (
            "Fichier d'arêtes (edges) fourni → graphe activé."
            if has_edges else "Pas de fichier d'arêtes → module relationnel désactivé (--edges <path>)."
        ),
    }

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Analyse du fichier d'arêtes
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_edges(edges_path: str) -> dict:
    try:
        df_e = _load(edges_path)
    except Exception as e:
        return {"error": str(e)}
    src_cols = [c for c in df_e.columns if "source" in c.lower() or "from" in c.lower() or "src" in c.lower()]
    dst_cols = [c for c in df_e.columns if "target" in c.lower() or "to" in c.lower() or "dst" in c.lower()]
    return {
        "n_edges":   len(df_e),
        "n_cols":    len(df_e.columns),
        "columns":   df_e.columns.tolist(),
        "src_col_guess": src_cols[0] if src_cols else None,
        "dst_col_guess": dst_cols[0] if dst_cols else None,
        "n_unique_src": int(df_e[src_cols[0]].nunique()) if src_cols else None,
        "n_unique_dst": int(df_e[dst_cols[0]].nunique()) if dst_cols else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────────────────────

TICK  = "✅"
CROSS = "❌"
WARN  = "⚠️ "
INFO  = "ℹ️ "
SEP   = "─" * 60


def _print_report(report: dict) -> None:
    r = report
    print()
    print("=" * 60)
    print("  🔍  RAPPORT D'INSPECTION DATASET — BotOrNot")
    print(f"  Généré le : {r['generated_at']}")
    print("=" * 60)

    # Fichier
    fi = r["file_info"]
    print(f"\n📁  Fichier    : {fi['path']}")
    print(f"    Format    : {fi['format'].upper()}")
    print(f"    Lignes    : {fi['n_rows']:,}")
    print(f"    Colonnes  : {fi['n_cols']}")

    # Colonnes avec manquants élevés
    high_miss = [c for c in r["columns"] if c["missing_%"] > 20]
    if high_miss:
        print(f"\n{WARN} Colonnes avec > 20% de valeurs manquantes :")
        for c in high_miss:
            print(f"    • {c['column']:<35} {c['missing_%']}% manquant")
    else:
        print(f"\n{TICK} Toutes les colonnes ont < 20% de valeurs manquantes.")

    # Signaux détectés
    sig = r["signals"]
    print(f"\n{SEP}")
    print("  SIGNAUX DÉTECTÉS")
    print(SEP)

    def _sig_line(label, info, key="found"):
        icon = TICK if info.get(key) else CROSS
        col  = f" → [{info.get('column','')}]" if info.get("column") else ""
        return f"  {icon} {label:<22}{col}"

    print(_sig_line("account_id", sig["account_id"]))
    if sig["account_id"]["found"]:
        print(f"       {sig['account_id']['n_unique']:,} comptes uniques")

    print(_sig_line("texte", sig["text"]))
    if sig["text"]["found"]:
        print(f"       longueur moy. {sig['text']['avg_length']} chars, "
              f"{sig['text']['empty_pct']}% vides")

    print(_sig_line("timestamps", sig["timestamps"]))
    if sig["timestamps"]["found"] and "min" in sig["timestamps"]:
        print(f"       {sig['timestamps']['min'][:10]} → "
              f"{sig['timestamps']['max'][:10]}, "
              f"granularité : {sig['timestamps'].get('granularity','?')}")

    print(_sig_line("label", sig["label"]))
    if sig["label"]["found"]:
        dist = sig["label"]["distribution"]
        dist_str = ", ".join(f"{k}={v:,}" for k, v in dist.items())
        print(f"       {dist_str}  [{sig['label']['balance']}]")

    prof = sig["profile_features"]
    print(_sig_line("profil (followers…)", prof))
    if prof["found"]:
        print(f"       {', '.join(prof['columns'].keys())}")

    # Graphe
    if "edges" in r:
        print(_sig_line("graphe (edges)", {"found": True, "column": r["edges"]["src_col_guess"]}))
        print(f"       {r['edges']['n_edges']:,} arêtes")
    else:
        print(f"  {CROSS} graphe (edges)        → non fourni (--edges <path>)")

    # Unité de prédiction
    print(f"\n{SEP}")
    print("  UNITÉ DE PRÉDICTION PROBABLE")
    print(SEP)
    print(f"  → {r['prediction_unit']}")

    # Recommandations
    print(f"\n{SEP}")
    print("  MODULES RECOMMANDÉS")
    print(SEP)
    for mod, rec in r["recommendations"].items():
        icon = TICK if rec["activate"] else CROSS
        status = "ACTIVER " if rec["activate"] else "désactivé"
        print(f"  {icon} {mod:<22} {status}")
        print(f"       {rec['reason']}")

    # Warnings
    if r.get("warnings"):
        print(f"\n{SEP}")
        print("  AVERTISSEMENTS")
        print(SEP)
        for w in r["warnings"]:
            print(f"  {WARN} {w}")

    print(f"\n{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Construction du rapport
# ─────────────────────────────────────────────────────────────────────────────

def inspect(data_path: str, edges_path: str = None) -> dict:
    t0 = datetime.now()

    df = _load(data_path)
    n_rows, n_cols = df.shape

    col_details = _analyze_columns(df)
    matches     = _match_cols(df.columns, PATTERNS)
    signals     = _detect_signals(df, matches)
    pred_unit   = _detect_prediction_unit(df, matches, col_details)
    has_edges   = edges_path is not None
    recs        = _recommend_modules(signals, has_edges, n_rows)

    # Avertissements
    warnings_list = []
    if not signals["account_id"]["found"]:
        warnings_list.append("Aucun account_id détecté — difficile de grouper les posts par compte.")
    if not signals["label"]["found"]:
        warnings_list.append("Aucun label détecté — mode inférence uniquement (pas d'entraînement possible).")
    if signals["label"]["found"] and signals["label"]["balance"] == "high_imbalance":
        warnings_list.append("Fort déséquilibre de classes — envisager class_weight='balanced'.")
    if n_rows > 90_000:
        warnings_list.append("Fichier tronqué à 100 000 premières lignes pour l'inspection rapide.")

    elapsed = (datetime.now() - t0).total_seconds()

    report = {
        "generated_at":    t0.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "file_info": {
            "path":   str(Path(data_path).resolve()),
            "format": Path(data_path).suffix.lstrip("."),
            "n_rows": n_rows,
            "n_cols": n_cols,
        },
        "columns":           col_details,
        "signals":           signals,
        "prediction_unit":   pred_unit,
        "recommendations":   recs,
        "warnings":          warnings_list,
    }

    if has_edges:
        report["edges"] = _analyze_edges(edges_path)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspecte un dataset rapidement pour le pipeline BotOrNot.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("data",   help="Chemin vers le fichier principal (csv/json/jsonl/parquet/tsv)")
    parser.add_argument("--edges", default=None, help="Chemin vers le fichier d'arêtes (optionnel)")
    parser.add_argument("--output", default=None, help="Sauvegarder le rapport JSON vers ce fichier")
    parser.add_argument("--json-only", action="store_true",
                        help="Affiche uniquement le JSON brut (pour intégration pipeline)")
    args = parser.parse_args()

    report = inspect(args.data, edges_path=args.edges)

    if args.json_only:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_report(report)
        print(f"⏱️  Inspection terminée en {report['elapsed_seconds']}s")

    if args.output:
        out_path = args.output if args.output.endswith(".json") else args.output + ".json"
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"📄  Rapport sauvegardé : {out_path}")

    return report


if __name__ == "__main__":
    main()
