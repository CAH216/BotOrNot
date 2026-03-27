# -*- coding: utf-8 -*-
"""
BotOrNot — CLI principal
Utilisation :
    python -m src.cli.main <commande> [options]

Commandes disponibles :
    inspect        Inspecter un fichier de donnees brutes
    build-features Construire les features a partir des donnees brutes
    train          Entrainer les modeles
    validate       Lancer la validation croisee
    predict        Generer des predictions
"""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Helpers visuels
# ---------------------------------------------------------------------------

BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

_BANNER_LINES = [
    "",
    "  BotOrNot — Pipeline de detection de bots v1.0",
    "  " + "=" * 46,
    "  Regle d'or : le doute profite a l'humain.",
    "",
]
BANNER = "\n".join(_BANNER_LINES)


def _print_banner() -> None:
    print(BANNER)


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {CYAN}→{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}", file=sys.stderr)


def _section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")


# ---------------------------------------------------------------------------
# Commande : inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    """
    Inspecte un fichier de donnees brutes via le loader + profiler officiels.
    """
    from src.data.loaders  import load_bundle
    from src.data.profiler import profile_bundle
    from src.data.schema   import LabelCols

    _section("INSPECT — Analyse du dataset")

    input_path = Path(args.input)
    if not input_path.exists():
        _err(f"Fichier introuvable : {input_path}")
        return 1

    _info(f"Fichier : {input_path.resolve()}")
    _info(f"Taille  : {input_path.stat().st_size / 1024:.1f} Ko")

    # --- Chargement ---
    try:
        t0 = time.time()
        bundle = load_bundle(input_path=input_path, mode=args.mode, nrows=args.nrows)
        elapsed = time.time() - t0
        _ok(f"Charge en {elapsed:.2f}s  (format : {bundle.source_format.upper()})")
    except Exception as exc:
        _err(f"Erreur de chargement : {exc}")
        return 1

    # --- Résumé du DataBundle ---
    _section("Structure du DataBundle")
    _info(f"Comptes (accounts_df) : {bundle.n_accounts:,} lignes")
    _info(f"Posts   (posts_df)    : {bundle.n_posts:,} lignes")
    _info(f"Edges   (edges_df)    : {'oui' if bundle.edges_df is not None else 'non'}")
    _info(f"Labels  (labels_df)   : {'oui' if bundle.labels_df is not None else 'non'}")

    # --- Aperçu colonnes accounts ---
    if bundle.accounts_df is not None and not bundle.accounts_df.empty:
        _section("Colonnes accounts_df")
        df = bundle.accounts_df
        for col in df.columns:
            n_null   = int(df[col].isna().sum())
            pct      = 100 * n_null / max(len(df), 1)
            dtype    = str(df[col].dtype)
            null_tag = f"  [{YELLOW}{pct:.0f}% null{RESET}]" if n_null > 0 else ""
            print(f"      {col:<35} ({dtype}){null_tag}")

    # --- Aperçu colonnes posts ---
    if bundle.posts_df is not None and not bundle.posts_df.empty:
        _section("Colonnes posts_df")
        df = bundle.posts_df
        for col in df.columns:
            n_null   = int(df[col].isna().sum())
            pct      = 100 * n_null / max(len(df), 1)
            dtype    = str(df[col].dtype)
            null_tag = f"  [{YELLOW}{pct:.0f}% null{RESET}]" if n_null > 0 else ""
            print(f"      {col:<35} ({dtype}){null_tag}")

    # --- Distribution des labels ---
    if bundle.labels_df is not None:
        _section("Distribution des labels")
        ldf    = bundle.labels_df
        counts = ldf[LabelCols.LABEL].value_counts()
        for val, cnt in counts.items():
            pct        = 100 * cnt / max(len(ldf), 1)
            label_name = "bot" if val == 1 else "humain"
            print(f"      {val:.0f} ({label_name}) -> {cnt:,}  ({pct:.1f}%)")

    # --- Profilage automatique via profiler.py ---
    try:
        profile = profile_bundle(bundle)
    except Exception as exc:
        _warn(f"Profilage echoue : {exc}")
        print()
        return 0

    _section("Flags du profiler (10 signaux)")
    flags = profile.to_dict()
    for flag, value in flags.items():
        if flag == "temporal_granularity":
            icon = f"{CYAN}i{RESET}"
            print(f"    [{icon}] {flag:<30} = {value}")
        else:
            icon = f"{GREEN}+{RESET}" if value else f"{RED}-{RESET}"
            print(f"    [{icon}] {flag:<30} = {value}")

    # --- Métriques complémentaires ---
    _section("Metriques")
    _info(f"Posts/compte (moyenne) : {profile.posts_per_account_mean}")
    _info(f"Posts/compte (max)     : {profile.posts_per_account_max}")
    if profile.class_balance:
        _info("Equilibre des classes  :")
        for label, info in profile.class_balance.items():
            name = "bot" if label == "1" else "humain"
            print(f"      {label} ({name}) : {info['count']:,}  ({info['pct']}%)")
    if profile.imbalance_ratio:
        ratio = profile.imbalance_ratio
        if ratio > 5:
            _warn(f"Desequilibre fort : {ratio:.1f}:1 → activer class_weight")
        else:
            _info(f"Ratio desequilibre     : {ratio:.1f}:1  (acceptable)")

    # --- Modules recommandés ---
    _section("Modules recommandes")
    for mod in profile.recommended_modules:
        _ok(f"ACTIVER  : {mod}")
    for mod in profile.disabled_modules:
        _warn(f"DESACTIVER : {mod}")

    # --- Avertissements ---
    if profile.warnings:
        _section("Avertissements")
        for w in profile.warnings:
            _warn(w)

    print()
    return 0



# ---------------------------------------------------------------------------
# Commande : build-features
# ---------------------------------------------------------------------------

def cmd_build_features(args: argparse.Namespace) -> int:
    _section("BUILD-FEATURES — Construction des features")

    input_path = Path(args.input)
    if not input_path.exists():
        _err(f"Fichier introuvable : {input_path}")
        return 1

    output_path = Path(args.output) if args.output else Path("data/processed/features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _info(f"Entrée  : {input_path}")
    _info(f"Sortie  : {output_path}")
    _warn("Module build-features : non encore implémenté (Mission 2+)")
    _info("Modules prévus : tabular / temporal / text_basic / structural / relational")

    print()
    return 0


# ---------------------------------------------------------------------------
# Commande : train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    _section("TRAIN — Entraînement des modèles")

    config_path = Path(args.config)
    if not config_path.exists():
        _err(f"Fichier de config introuvable : {config_path}")
        return 1

    _info(f"Config : {config_path}")
    _warn("Module train : non encore implémenté (Mission 3+)")
    _info("Modèles prévus : LightGBM → CatBoost → XGBoost → Ensemble")

    print()
    return 0


# ---------------------------------------------------------------------------
# Commande : validate
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    _section("VALIDATE — Validation croisée")

    config_path = Path(args.config)
    if not config_path.exists():
        _err(f"Fichier de config introuvable : {config_path}")
        return 1

    _info(f"Config : {config_path}")
    _warn("Module validate : non encore implémenté (Mission 3+)")
    _info("CV prévue : StratifiedKFold / GroupKFold (anti-leakage par compte)")

    print()
    return 0


# ---------------------------------------------------------------------------
# Commande : predict
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> int:
    _section("PREDICT — Génération des prédictions")

    input_path = Path(args.input)
    if not input_path.exists():
        _err(f"Fichier introuvable : {input_path}")
        return 1

    model_path = Path(args.model) if args.model else None
    if model_path and not model_path.exists():
        _err(f"Modèle introuvable : {model_path}")
        return 1

    output_path = Path(args.output) if args.output else Path("data/submissions/predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _info(f"Entrée  : {input_path}")
    _info(f"Modèle  : {model_path or '(non spécifié)'}")
    _info(f"Sortie  : {output_path}")
    _warn("Module predict : non encore implémenté (Mission 4+)")
    _info("Sortie prévue : account_id, bot_probability, label")

    print()
    return 0


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.main",
        description="BotOrNot — Pipeline de détection de bots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m src.cli.main inspect        --input data/raw/train.csv
  python -m src.cli.main build-features --input data/raw/train.csv --output data/processed/features.parquet
  python -m src.cli.main train          --config configs/default.yaml
  python -m src.cli.main validate       --config configs/default.yaml
  python -m src.cli.main predict        --input data/raw/test.csv --model artifacts/models/best.pkl
        """,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<commande>")
    subparsers.required = True

    # --- inspect ---
    p_inspect = subparsers.add_parser(
        "inspect",
        help="Inspecter un fichier de donnees et profiler le dataset",
    )
    p_inspect.add_argument("--input",  required=True, help="Chemin vers le fichier brut (CSV/JSON/JSONL)")
    p_inspect.add_argument("--nrows",  type=int, default=None, help="Nombre de lignes a charger (defaut: toutes)")
    p_inspect.add_argument("--mode",   default="auto",
                           choices=["auto", "accounts", "posts"],
                           help="Mode de lecture : auto (defaut) | accounts | posts")
    p_inspect.set_defaults(func=cmd_inspect)

    # --- build-features ---
    p_bf = subparsers.add_parser(
        "build-features",
        help="Construire les features à partir des données brutes",
    )
    p_bf.add_argument("--input",  required=True, help="Fichier brut d'entrée")
    p_bf.add_argument("--output", default=None,  help="Fichier de features de sortie (.parquet)")
    p_bf.add_argument("--config", default="configs/features.yaml", help="Config features")
    p_bf.set_defaults(func=cmd_build_features)

    # --- train ---
    p_train = subparsers.add_parser(
        "train",
        help="Entraîner les modèles sur les features construites",
    )
    p_train.add_argument("--config", default="configs/default.yaml", help="Config principale")
    p_train.set_defaults(func=cmd_train)

    # --- validate ---
    p_val = subparsers.add_parser(
        "validate",
        help="Lancer la validation croisée (anti-leakage)",
    )
    p_val.add_argument("--config", default="configs/default.yaml", help="Config principale")
    p_val.set_defaults(func=cmd_validate)

    # --- predict ---
    p_pred = subparsers.add_parser(
        "predict",
        help="Générer des prédictions sur de nouvelles données",
    )
    p_pred.add_argument("--input",  required=True, help="Fichier de données à prédire")
    p_pred.add_argument("--model",  default=None,  help="Chemin vers le modèle sauvegardé (.pkl)")
    p_pred.add_argument("--output", default=None,  help="Chemin de sortie (CSV)")
    p_pred.set_defaults(func=cmd_predict)

    return parser


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    _print_banner()
    parser = build_parser()
    args   = parser.parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
