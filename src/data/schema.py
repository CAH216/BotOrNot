# -*- coding: utf-8 -*-
"""
src/data/schema.py
------------------
Contrats de colonnes internes du pipeline BotOrNot.

Le reste du pipeline ne doit JAMAIS lire les colonnes brutes directement.
Il ne travaille qu'avec les noms canoniques définis ici.

Structure interne normalisee :
    accounts_df  — une ligne par compte
    posts_df     — une ligne par publication
    edges_df     — (optionnel) une ligne par relation inter-comptes
    labels_df    — (optionnel) une ligne par compte avec son label
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Noms de colonnes canoniques
# ---------------------------------------------------------------------------

class AccountCols:
    """Colonnes canoniques de accounts_df."""
    ID             = "account_id"       # identifiant unique du compte
    CREATED_AT     = "created_at"       # datetime de creation du compte (Timestamp ou NaT)
    SCREEN_NAME    = "screen_name"      # nom d'utilisateur / handle
    BIO            = "bio"              # description / bio du profil (str ou "")
    FOLLOWERS      = "followers_count"  # nombre de followers
    FOLLOWING      = "following_count"  # nombre de comptes suivis
    TOTAL_POSTS    = "statuses_count"   # nb total de publications
    VERIFIED       = "verified"         # bool ou NaN
    PROFILE_IMAGE  = "default_profile_image"  # bool : image par defaut ?
    LOCATION       = "location"         # localisation textuelle
    LANG           = "lang"             # langue du compte


class PostCols:
    """Colonnes canoniques de posts_df."""
    ID         = "post_id"      # identifiant unique du post
    ACCOUNT_ID = "account_id"   # FK vers accounts_df
    TEXT       = "text"         # contenu brut du post
    TEXT_CLEAN = "text_clean"   # version nettoyee (remplie par preprocessing)
    CREATED_AT = "created_at"   # datetime de publication
    LANG       = "lang"         # langue detectee / declaree
    SOURCE     = "source"       # client / API utilise
    IN_REPLY_TO = "in_reply_to_account_id"  # si c'est une reponse
    RETWEET_OF  = "retweet_of_account_id"   # si c'est un retweet
    HASHTAGS    = "hashtags"    # list[str] (extraite ou brute)
    MENTIONS    = "mentions"    # list[str]
    URLS        = "urls"        # list[str]


class EdgeCols:
    """Colonnes canoniques de edges_df."""
    SOURCE  = "source_account_id"
    TARGET  = "target_account_id"
    TYPE    = "edge_type"     # "follow" | "reply" | "retweet" | "mention"
    WEIGHT  = "weight"        # nb d'interactions (optionnel)


class LabelCols:
    """Colonnes canoniques de labels_df."""
    ACCOUNT_ID = "account_id"
    LABEL      = "label"      # 1 = bot, 0 = humain
    CONFIDENCE = "confidence" # optionnel — niveau de confiance du label


# ---------------------------------------------------------------------------
# Alias de colonnes brutes → canoniques
# (utilise par normalize_columns dans preprocessing)
# ---------------------------------------------------------------------------

ACCOUNT_ID_ALIASES: List[str] = [
    "user_id", "account_id", "author_id", "userid",
    "user", "uid", "id_user",
]

POST_ID_ALIASES: List[str] = [
    "post_id", "tweet_id", "id", "status_id", "postid",
]

TEXT_ALIASES: List[str] = [
    "text", "tweet", "content", "body", "post", "message",
    "full_text", "tweet_text",
]

CREATED_AT_ALIASES: List[str] = [
    "created_at", "timestamp", "post_time", "date", "datetime",
    "created", "time", "posted_at",
]

SCREEN_NAME_ALIASES: List[str] = [
    "screen_name", "username", "handle", "name", "login",
]

BIO_ALIASES: List[str] = [
    "bio", "description", "profile_description", "about",
]

FOLLOWERS_ALIASES: List[str] = [
    "followers_count", "followers", "nb_followers",
]

FOLLOWING_ALIASES: List[str] = [
    "following_count", "following", "friends_count", "nb_following",
]

TOTAL_POSTS_ALIASES: List[str] = [
    "statuses_count", "total_posts", "tweet_count", "nb_posts",
    "post_count", "tweets",
]

LABEL_ALIASES: List[str] = [
    "label", "bot", "is_bot", "class", "target", "y",
    "bot_label", "account_type",
]

SOURCE_ALIASES: List[str] = [
    "source", "client", "api", "app", "platform",
]

# Dictionnaire global alias → colonne canonique (pour loaders)
ALIAS_MAP = {
    **{a: AccountCols.ID          for a in ACCOUNT_ID_ALIASES},
    **{a: PostCols.ID             for a in POST_ID_ALIASES},
    **{a: PostCols.TEXT           for a in TEXT_ALIASES},
    **{a: PostCols.CREATED_AT     for a in CREATED_AT_ALIASES},
    **{a: AccountCols.SCREEN_NAME for a in SCREEN_NAME_ALIASES},
    **{a: AccountCols.BIO         for a in BIO_ALIASES},
    **{a: AccountCols.FOLLOWERS   for a in FOLLOWERS_ALIASES},
    **{a: AccountCols.FOLLOWING   for a in FOLLOWING_ALIASES},
    **{a: AccountCols.TOTAL_POSTS for a in TOTAL_POSTS_ALIASES},
    **{a: LabelCols.LABEL         for a in LABEL_ALIASES},
    **{a: PostCols.SOURCE         for a in SOURCE_ALIASES},
}


# ---------------------------------------------------------------------------
# Structure de retour du loader
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """
    Conteneur principal du pipeline.
    Toutes les couches du pipeline recoivent et retournent un DataBundle.
    """
    accounts_df: object = None   # pd.DataFrame | None
    posts_df:    object = None   # pd.DataFrame | None
    edges_df:    object = None   # pd.DataFrame | None  (optionnel)
    labels_df:   object = None   # pd.DataFrame | None  (optionnel)

    # Flags de disponibilite (remplis par le profiler)
    flags: dict = field(default_factory=dict)

    # Metadata de chargement
    source_path:  str = ""
    source_format: str = ""    # "csv" | "json" | "jsonl" | "multi"
    n_accounts:   int = 0
    n_posts:      int = 0

    def summary(self) -> str:
        lines = [
            f"DataBundle — {self.source_format.upper()} — {self.source_path}",
            f"  accounts : {self.n_accounts:,}",
            f"  posts    : {self.n_posts:,}",
            f"  edges    : {'oui' if self.edges_df is not None else 'non'}",
            f"  labels   : {'oui' if self.labels_df is not None else 'non'}",
        ]
        if self.flags:
            lines.append("  flags    :")
            for k, v in self.flags.items():
                icon = "+" if v else "-"
                lines.append(f"    [{icon}] {k}")
        return "\n".join(lines)
