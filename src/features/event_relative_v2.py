# -*- coding: utf-8 -*-
"""
src/features/event_relative_v2.py
Exploit Every Bit V2 : Bloc de features event-relative et metadata-aware.

Familles couvertes :
1. tweet_count / users_average_amount_posts
2. z_score residual = z_score - users_average_z_score
3. first_post_offset / last_post_offset dans la fenêtre [start_time, end_time]
4. densité d'activité relative à la fenêtre
5. topic alignment : proportion de posts collant aux keywords/topics de l'event
6. topic overfocus / underfocus (indicateurs de Persona)
7. contradiction lang event ↔ posts/bio
8. contradictions croisées name / username / description / location
"""
import re
import pandas as pd
import numpy as np
from datetime import timezone


# ─── helpers ───────────────────────────────────────────────────────────────

def _parse_utc(series: pd.Series) -> pd.Series:
    """Convertit une Series de strings ISO en datetime UTC sans erreur."""
    return pd.to_datetime(series, errors="coerce", utc=True)

def _safe_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)

def _lang_guess(text: str) -> str:
    """Heuristique légère : détecte EN/FR en cherchant stopwords fréquents."""
    t = text.lower()
    fr_hits = sum(1 for w in ["le ", "la ", "les ", "de ", "du ", "un ", "une ", "est ", "et ", "pas "] if w in t)
    en_hits = sum(1 for w in ["the ", " is ", " are ", " a ", " an ", " of ", " it ", " for ", " in "] if w in t)
    if fr_hits > en_hits:
        return "fr"
    elif en_hits > fr_hits:
        return "en"
    return "unk"

def _char_overlap(s1: str, s2: str) -> float:
    """Jaccard sur caractères uniques. Retourne 0 si l'un est vide."""
    if not s1 or not s2:
        return 0.0
    a, b = set(s1), set(s2)
    return len(a & b) / len(a | b) if (a | b) else 0.0

def _n_digits(s: str) -> int:
    return sum(c.isdigit() for c in s)


# ─── extracteur principal ───────────────────────────────────────────────────

def extract_event_relative_v2(
    u_df: pd.DataFrame,
    p_df: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """
    Retourne un DataFrame indexé sur user_id avec toutes les features
    de la famille "Exploit Every Bit V2".
    Aucune heuristique de classification : tout est numérique/binaire pour LightGBM.
    """
    if u_df.empty:
        return pd.DataFrame()

    uid_col = "user_id"
    res = pd.DataFrame({uid_col: u_df[uid_col].values})

    # ── pull metadata ────────────────────────────────────────────────────────
    avg_posts  = float(metadata.get("users_average_amount_posts", 1.0)) or 1.0
    avg_z      = float(metadata.get("users_average_z_score", 0.0))
    start_time = metadata.get("start_time", None)
    end_time   = metadata.get("end_time", None)

    topics = metadata.get("topics", [])
    keywords: set[str] = set()
    topic_names: set[str] = set()
    for t in topics:
        topic_names.add(t.get("topic", "").lower())
        for kw in t.get("keywords", []):
            keywords.add(kw.lower().strip())

    # Déduis la "langue officielle de l'event" depuis les keywords (heuristique propre)
    kw_sample = " ".join(list(keywords)[:30])
    event_lang = _lang_guess(kw_sample)

    # ── 1. Ratios tabulaires normalisés ─────────────────────────────────────
    tc = u_df.get("tweet_count", pd.Series([0] * len(u_df), index=u_df.index)).fillna(0).values.astype(float)
    zs = u_df.get("z_score", pd.Series([0.0] * len(u_df), index=u_df.index)).fillna(0.0).values.astype(float)

    res["ev_tweet_ratio"]      = tc / avg_posts          # >1 = sur-actif vs event
    res["ev_z_score_residual"] = zs - avg_z              # >0 = z anormalement haut vs event

    # ── 2. Offsets temporels dans la fenêtre ─────────────────────────────────
    try:
        t_start = pd.Timestamp(start_time, tz="UTC") if start_time else None
        t_end   = pd.Timestamp(end_time,   tz="UTC") if end_time   else None
        window_sec = (t_end - t_start).total_seconds() if (t_start and t_end) else None
    except Exception:
        t_start = t_end = window_sec = None

    if not p_df.empty and "created_at" in p_df.columns and window_sec and window_sec > 0:
        p_ts = p_df[["user_id", "created_at"]].copy()
        p_ts["dt"] = _parse_utc(p_ts["created_at"])
        p_ts = p_ts.dropna(subset=["dt"])

        if not p_ts.empty:
            # Offset premier / dernier post depuis start_time (en fraction de fenêtre)
            p_ts["offset_from_start"] = (p_ts["dt"] - t_start).dt.total_seconds() / window_sec
            p_ts["offset_from_start"] = p_ts["offset_from_start"].clip(0, 1)

            first_last = p_ts.groupby("user_id").agg(
                ev_first_post_offset=("offset_from_start", "min"),
                ev_last_post_offset=("offset_from_start", "max"),
                ev_post_span=("offset_from_start", lambda x: x.max() - x.min()),
                ev_post_count_in_window=("dt", "count"),
            ).reset_index()

            # densité : posts par heure dans la fenêtre
            window_h = window_sec / 3600
            first_last["ev_density_per_hour"] = first_last["ev_post_count_in_window"] / window_h

            res = res.merge(first_last, on="user_id", how="left")
        else:
            for c in ["ev_first_post_offset", "ev_last_post_offset", "ev_post_span",
                      "ev_post_count_in_window", "ev_density_per_hour"]:
                res[c] = 0.0
    else:
        for c in ["ev_first_post_offset", "ev_last_post_offset", "ev_post_span",
                  "ev_post_count_in_window", "ev_density_per_hour"]:
            res[c] = 0.0

    # ── 3. Topic alignment ───────────────────────────────────────────────────
    if not p_df.empty and "text" in p_df.columns and keywords:
        p_kw = p_df[["user_id", "text"]].copy()
        p_kw["text_l"] = _safe_str(p_kw["text"]).str.lower()

        def has_kw(txt: str) -> int:
            return int(any(kw in txt for kw in keywords))

        p_kw["has_kw"] = p_kw["text_l"].apply(has_kw)
        topic_agg = p_kw.groupby("user_id").agg(
            ev_topic_focus=("has_kw", "mean"),
        ).reset_index()
        res = res.merge(topic_agg, on="user_id", how="left")
    else:
        res["ev_topic_focus"] = 0.0

    res["ev_topic_focus"] = res["ev_topic_focus"].fillna(0.0)
    # Overfocus (100%) = probable bot amplifiant le sujet ; 0% = off-topic ou ghost
    res["ev_topic_overfocus"]  = (res["ev_topic_focus"] >= 0.95).astype(np.float32)
    res["ev_topic_underfocus"] = (res["ev_topic_focus"] == 0.0).astype(np.float32)

    # ── 4. Contradiction lang event ↔ posts ──────────────────────────────────
    if not p_df.empty and "text" in p_df.columns:
        p_lg = p_df[["user_id", "text"]].copy()
        p_lg["text"] = _safe_str(p_lg["text"])

        p_lg["uses_french"] = p_lg["text"].apply(
            lambda t: int(any(w in t.lower() for w in ["le ", "la ", "les ", "de ", " est ", "et ", "pas ", "que "]))
        )
        p_lg["uses_english"] = p_lg["text"].apply(
            lambda t: int(any(w in t.lower() for w in ["the ", " is ", " are ", " of ", " it ", " in ", " for "]))
        )
        lang_agg = p_lg.groupby("user_id").agg(
            ev_frac_french=("uses_french", "mean"),
            ev_frac_english=("uses_english", "mean"),
        ).reset_index()
        res = res.merge(lang_agg, on="user_id", how="left").fillna(0.0)
    else:
        res["ev_frac_french"]  = 0.0
        res["ev_frac_english"] = 0.0

    # Contradiction : event EN mais compte poste FR majoritairement (et vice versa)
    if event_lang == "en":
        res["ev_lang_contradiction"] = res["ev_frac_french"]
    elif event_lang == "fr":
        res["ev_lang_contradiction"] = res["ev_frac_english"]
    else:
        res["ev_lang_contradiction"] = 0.0

    # Contradiction lang event ↔ bio
    bio = _safe_str(u_df.get("description", pd.Series([""] * len(u_df), index=u_df.index)))
    bio_lang = bio.apply(_lang_guess)
    if event_lang in ("en", "fr"):
        res["ev_bio_lang_mismatch"] = (bio_lang != event_lang).astype(np.float32)
    else:
        res["ev_bio_lang_mismatch"] = 0.0

    # ── 5. Contradictions croisées profil ─────────────────────────────────────
    username = _safe_str(u_df.get("username", pd.Series([""] * len(u_df), index=u_df.index))).str.lower()
    name     = _safe_str(u_df.get("name",     pd.Series([""] * len(u_df), index=u_df.index))).str.lower()
    desc     = bio.str.lower()
    location = _safe_str(u_df.get("location", pd.Series([""] * len(u_df), index=u_df.index))).str.lower()

    # Overlap username ↔ name (bots ont souvent zéro overlap random + suffix digits)
    res["ev_name_usr_overlap"] = [
        _char_overlap(u, n) for u, n in zip(username.values, name.values)
    ]

    # Proportion de chiffres dans le username (forte chez les bots générés)
    res["ev_usr_digit_ratio"] = [
        _n_digits(u) / max(len(u), 1) for u in username.values
    ]

    # Bio ↔ Location : bots parfois copie-collent la bio dans la location
    res["ev_bio_loc_overlap"] = [
        _char_overlap(b, l) for b, l in zip(desc.values, location.values)
    ]

    # Longueur description vs username : bots avec bio vide mais username long
    usr_lens  = username.str.len().values.astype(float)
    desc_lens = desc.str.len().values.astype(float)
    res["ev_desc_empty_usr_long"] = (
        (desc_lens == 0) & (usr_lens > 10)
    ).astype(np.float32)

    # Location présente mais bio absente
    loc_lens = location.str.len().values.astype(float)
    res["ev_loc_no_bio"] = (
        (loc_lens > 0) & (desc_lens == 0)
    ).astype(np.float32)

    # ── nettoyage final ──────────────────────────────────────────────────────
    float_cols = [c for c in res.columns if c != uid_col]
    for c in float_cols:
        res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0.0)

    return res
