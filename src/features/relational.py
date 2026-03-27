# -*- coding: utf-8 -*-
"""
src/features/relational.py
---------------------------
Extracteur de features relationnelles — graphe de relations entre comptes.

Principe :
    Ce module s'auto-désactive proprement si edges_df est absent ou vide.
    Il ne requiert PAS networkx : toutes les métriques sont calculées en
    pandas/numpy pur. networkx est utilisé en option pour le clustering
    coefficient si disponible, sinon une approximation locale est utilisée.

Signaux extraits :
    Degré           — in-degree, out-degree, degré total
    Réciprocité     — fraction de liens réciproques
    Clustering      — coefficient de clustering local (ou approximation)
    Composante      — taille de la composante connectée
    Synchronisation — fraction de voisins avec une activité temporelle proche

Usage :
    from src.features.relational import extract_relational_features

    feat_df = extract_relational_features(edges_df, accounts_df, posts_df)
    # Retourne DataFrame vide si edges_df est absent/vide
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import AccountCols

logger = logging.getLogger(__name__)

_PREFIX = "rel_"

RELATIONAL_COLS = [
    "rel_in_degree",               # nb de liens entrants
    "rel_out_degree",              # nb de liens sortants
    "rel_total_degree",            # degré total
    "rel_reciprocity",             # fraction de liens réciproques
    "rel_clustering_approx",       # coefficient de clustering approximatif
    "rel_component_size",          # taille de la composante connexe
    "rel_component_size_log",      # log(taille) moins sensible aux outliers
    "rel_neighbor_sync_score",     # synchronisation temporelle avec les voisins
    "rel_in_out_ratio",            # in/out ratio (masse follower relative)
]


# ---------------------------------------------------------------------------
# Auto-désactivation
# ---------------------------------------------------------------------------

def _is_empty(edges_df: Optional[pd.DataFrame]) -> bool:
    """Retourne True si edges_df est absent ou vide."""
    return (
        edges_df is None
        or not isinstance(edges_df, pd.DataFrame)
        or edges_df.empty
        or len(edges_df.columns) < 2
    )


def _detect_edge_cols(edges_df: pd.DataFrame) -> tuple[str, str]:
    """
    Détecte automatiquement les colonnes source/target.
    Cherche des colonnes avec 'source','from','src','follower' et
    'target','to','dst','following'.
    """
    src_candidates = ["source", "from", "src", "follower", "account_id"]
    tgt_candidates = ["target", "to", "dst", "following", "target_id"]

    src_col = next((c for c in edges_df.columns
                    if any(k in c.lower() for k in src_candidates)), edges_df.columns[0])
    tgt_col = next((c for c in edges_df.columns
                    if any(k in c.lower() for k in tgt_candidates)
                    and c != src_col), edges_df.columns[1])
    return src_col, tgt_col


# ---------------------------------------------------------------------------
# Metrics pandas/numpy
# ---------------------------------------------------------------------------

def _compute_degrees(
    edges_df: pd.DataFrame,
    src_col:  str,
    tgt_col:  str,
) -> pd.DataFrame:
    """Calcule in-degree, out-degree, total degree pour chaque nœud."""
    out_deg = edges_df.groupby(src_col).size().rename("rel_out_degree")
    in_deg  = edges_df.groupby(tgt_col).size().rename("rel_in_degree")

    degree_df = (
        pd.concat([out_deg, in_deg], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"index": AccountCols.ID, src_col: AccountCols.ID})
    )

    # Certains nœuds n'apparaissent que d'un côté
    all_nodes = set(edges_df[src_col]) | set(edges_df[tgt_col])
    degree_df = degree_df.reindex(
        pd.RangeIndex(len(all_nodes))
    )

    # Rebuild proprement
    out_s = edges_df.groupby(src_col).size()
    in_s  = edges_df.groupby(tgt_col).size()
    all_nodes_series = pd.Series(sorted(all_nodes))

    df = pd.DataFrame({AccountCols.ID: all_nodes_series})
    df = df.join(out_s.rename("rel_out_degree"), on=AccountCols.ID)
    df = df.join(in_s.rename("rel_in_degree"),   on=AccountCols.ID)
    df["rel_out_degree"] = df["rel_out_degree"].fillna(0).astype(int)
    df["rel_in_degree"]  = df["rel_in_degree"].fillna(0).astype(int)
    df["rel_total_degree"] = df["rel_out_degree"] + df["rel_in_degree"]
    df["rel_in_out_ratio"] = (
        df["rel_in_degree"] / (df["rel_out_degree"] + 1)
    ).round(4)

    return df.reset_index(drop=True)


def _compute_reciprocity(
    edges_df: pd.DataFrame,
    src_col:  str,
    tgt_col:  str,
) -> pd.Series:
    """
    Calcule la réciprocité locale : fraction de voisins sortants qui
    ont un lien retour vers ce nœud.
    """
    # Créer un ensemble de paires (src, tgt)
    edge_set = set(zip(edges_df[src_col], edges_df[tgt_col]))

    # Pour chaque nœud source, compter les liens réciproques
    recip_data = {}
    for (s, t) in edge_set:
        reversed_exists = int((t, s) in edge_set)
        if s not in recip_data:
            recip_data[s] = []
        recip_data[s].append(reversed_exists)

    recip_series = pd.Series(
        {node: round(np.mean(vals), 4) for node, vals in recip_data.items()},
        name="rel_reciprocity",
    )
    return recip_series


def _compute_clustering_approx(
    edges_df: pd.DataFrame,
    src_col:  str,
    tgt_col:  str,
) -> pd.Series:
    """
    Approximation locale du coefficient de clustering.

    Pour chaque nœud u :
        - Trouver ses voisins (union in+out)
        - Compter les liens existants entre ces voisins
        - Normaliser par k*(k-1)

    C'est une approximation O(k²) per node, pas O(n²) global.
    Limite à 30 voisins max pour les comptes très connectés.
    """
    MAX_NEIGHBORS = 30

    # Construire adjacency dict
    adj: dict = {}
    for _, row in edges_df.iterrows():
        s, t = row[src_col], row[tgt_col]
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)

    coeffs = {}
    for node, neighbors in adj.items():
        k = len(neighbors)
        if k < 2:
            coeffs[node] = 0.0
            continue
        # Limiter pour les hubs
        sample = list(neighbors)[:MAX_NEIGHBORS]
        sample_set = set(sample)
        k_s = len(sample)
        triangles = 0
        for nb in sample:
            nb_neighbors = adj.get(nb, set())
            triangles += len(nb_neighbors & sample_set) - int(node in nb_neighbors)
        max_triangles = k_s * (k_s - 1)
        coeffs[node] = round(triangles / max(max_triangles, 1), 4)

    return pd.Series(coeffs, name="rel_clustering_approx")


def _compute_components(
    edges_df: pd.DataFrame,
    src_col:  str,
    tgt_col:  str,
) -> pd.Series:
    """
    Calcule la taille de la composante connexe de chaque nœud
    via Union-Find (Disjoint Set Union — DSU).
    """
    parent: dict = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), x)
            x = parent.get(x, x)
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    all_nodes = set(edges_df[src_col]) | set(edges_df[tgt_col])
    for node in all_nodes:
        parent[node] = node

    for _, row in edges_df.iterrows():
        union(row[src_col], row[tgt_col])

    # Compter la taille de chaque composante
    from collections import Counter
    roots   = {node: find(node) for node in all_nodes}
    sizes   = Counter(roots.values())
    comp_sizes = {node: sizes[root] for node, root in roots.items()}

    return pd.Series(comp_sizes, name="rel_component_size")


def _compute_neighbor_sync(
    edges_df:  pd.DataFrame,
    posts_df:  Optional[pd.DataFrame],
    src_col:   str,
    tgt_col:   str,
    window_h:  float = 1.0,
) -> pd.Series:
    """
    Score de synchronisation temporelle avec les voisins.

    Pour chaque nœud u, regarde si ses voisins ont posté
    dans la même fenêtre temporelle que lui.

    Un score élevé = beaucoup de coordination temporelle → signal de bot.
    """
    if (posts_df is None or posts_df.empty
            or "created_at" not in posts_df.columns
            or AccountCols.ID not in posts_df.columns):
        return pd.Series(dtype=float, name="rel_neighbor_sync_score")

    ts_col = "created_at"
    try:
        posts_copy = posts_df.copy()
        posts_copy[ts_col] = pd.to_datetime(posts_copy[ts_col], utc=True, errors="coerce")
    except Exception:
        return pd.Series(dtype=float, name="rel_neighbor_sync_score")

    # Fenêtre en nanosecondes
    window_ns = int(window_h * 3600 * 1e9)

    # Résumé : premier timestamp par compte
    first_post = (posts_copy.groupby(AccountCols.ID)[ts_col]
                  .min()
                  .dropna()
                  .dt.value_counts())

    # Pour chaque nœud source
    adj: dict = {}
    for _, row in edges_df.iterrows():
        adj.setdefault(row[src_col], set()).add(row[tgt_col])

    first_posts = posts_copy.groupby(AccountCols.ID)[ts_col].min().dropna()
    sync_scores = {}

    for node, neighbors in adj.items():
        if node not in first_posts.index:
            sync_scores[node] = 0.0
            continue
        node_ts = first_posts[node].value
        nb_ts   = first_posts.reindex(list(neighbors)).dropna()
        if nb_ts.empty:
            sync_scores[node] = 0.0
            continue
        nb_vals = nb_ts.values.astype(np.int64)
        sync = float((np.abs(nb_vals - node_ts) <= window_ns).mean())
        sync_scores[node] = round(sync, 4)

    return pd.Series(sync_scores, name="rel_neighbor_sync_score")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def extract_relational_features(
    edges_df:    Optional[pd.DataFrame],
    accounts_df: Optional[pd.DataFrame] = None,
    posts_df:    Optional[pd.DataFrame] = None,
    sync_window_h: float = 1.0,
) -> pd.DataFrame:
    """
    Extrait les features relationnelles pour chaque compte.

    S'AUTO-DÉSACTIVE si edges_df est absent ou vide.

    Args:
        edges_df      : DataFrame de relations (source_id, target_id)
        accounts_df   : optionnel — pour créer des lignes NaN manquantes
        posts_df      : optionnel — pour la synchronisation temporelle
        sync_window_h : fenêtre de synchronisation en heures

    Returns:
        DataFrame avec account_id + features relationnelles.
        DataFrame VIDE (0 lignes) si edges_df absent.
    """
    id_col = AccountCols.ID

    if _is_empty(edges_df):
        logger.info("[relational] edges_df absent → module désactivé")
        return pd.DataFrame(columns=[id_col])

    try:
        src_col, tgt_col = _detect_edge_cols(edges_df)
    except Exception as e:
        logger.warning("[relational] Impossible de détecter src/tgt cols : %s", e)
        return pd.DataFrame(columns=[id_col])

    logger.info(
        "[relational] %d edges, cols=(%s, %s)",
        len(edges_df), src_col, tgt_col,
    )

    # 1. Degrees
    try:
        df = _compute_degrees(edges_df, src_col, tgt_col)
    except Exception as e:
        logger.error("[relational] Erreur degrees : %s", e)
        return pd.DataFrame(columns=[id_col])

    # 2. Réciprocité
    try:
        recip = _compute_reciprocity(edges_df, src_col, tgt_col)
        df = df.join(recip, on=id_col)
    except Exception as e:
        logger.warning("[relational] Réciprocité échouée : %s", e)
        df["rel_reciprocity"] = np.nan

    # 3. Clustering approximatif
    try:
        clust = _compute_clustering_approx(edges_df, src_col, tgt_col)
        df = df.join(clust, on=id_col)
    except Exception as e:
        logger.warning("[relational] Clustering échoué : %s", e)
        df["rel_clustering_approx"] = np.nan

    # 4. Taille des composantes
    try:
        comp = _compute_components(edges_df, src_col, tgt_col)
        df = df.join(comp, on=id_col)
        df["rel_component_size_log"] = np.log1p(
            df["rel_component_size"].fillna(1)
        ).round(4)
    except Exception as e:
        logger.warning("[relational] Composantes échouées : %s", e)
        df["rel_component_size"]     = np.nan
        df["rel_component_size_log"] = np.nan

    # 5. Synchronisation temporelle (optionnel)
    try:
        sync = _compute_neighbor_sync(edges_df, posts_df, src_col, tgt_col, sync_window_h)
        df = df.join(sync, on=id_col)
    except Exception as e:
        logger.warning("[relational] Sync temporelle échouée : %s", e)
        df["rel_neighbor_sync_score"] = np.nan

    # Remplir NaN pour les comptes sans données de synchronisation ou réciprocité
    for col in RELATIONAL_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Comptes dans accounts_df mais hors du graphe = 0 degré
    if accounts_df is not None and id_col in accounts_df.columns:
        in_graph = set(df[id_col])
        missing  = set(accounts_df[id_col]) - in_graph
        if missing:
            null_rows = pd.DataFrame({id_col: list(missing)})
            for col in RELATIONAL_COLS:
                null_rows[col] = 0.0
            df = pd.concat([df, null_rows], ignore_index=True)

    # Typage numérique
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)

    logger.info(
        "[relational] %d nœuds × %d features",
        len(df), len(df.columns) - 1
    )
    return df.reset_index(drop=True)
