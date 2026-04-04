#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_final_outputs.py
========================
Script de vérification des fichiers de soumission avant envoi.

Usage :
  python scripts/verify_final_outputs.py \\
      --en  dataset/dataset.posts&users.NEW_EN.json \\
      --fr  dataset/dataset.posts&users.NEW_FR.json \\
      --team MonTeam

Vérifie :
  ✓ Fichiers existent
  ✓ Encodage UTF-8 valide
  ✓ Un ID par ligne, aucune ligne vide
  ✓ Aucun doublon
  ✓ Aucun ID hors dataset
  ✓ Noms de fichiers corrects
  ✓ Ratio bots/total plausible
  ✓ IDs sont des strings non vides
"""

import argparse
import json
import sys
from pathlib import Path


def load_user_ids(json_path: str) -> set:
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    users = d.get("users", [])
    ids = set()
    for u in users:
        uid = str(u.get("id", "")).strip()
        if uid:
            ids.add(uid)
    return ids


def check_file(file_path: str, dataset_json: str, lang: str) -> tuple:
    """Vérifie un fichier de soumission. Retourne (ok: bool, issues: list, stats: dict)."""
    issues = []
    warnings_list = []
    path = Path(file_path)

    # 1. Existence
    if not path.exists():
        return False, [f"❌ CRITIQUE : Fichier introuvable : {file_path}"], {}

    # 2. Nom correct
    expected_suffix = f".detections.{lang.lower()}.txt"
    if not path.name.endswith(expected_suffix):
        issues.append(f"⚠  Nom fichier inattendu : '{path.name}' (attendu: *{expected_suffix})")

    # 3. Lecture UTF-8
    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return False, [f"❌ CRITIQUE : Encodage non-UTF-8 : {e}"], {}

    # 4. Lignes
    lines_raw = content.split("\n")
    # Supprimer dernière ligne vide si fichier se termine par \n
    if lines_raw and lines_raw[-1] == "":
        lines_raw = lines_raw[:-1]

    if not lines_raw:
        return False, ["❌ CRITIQUE : Fichier vide (0 IDs)"], {}

    # 5. Lignes vides internes
    empty_lines = [i + 1 for i, l in enumerate(lines_raw) if l.strip() == ""]
    if empty_lines:
        issues.append(f"❌ Lignes vides aux positions : {empty_lines[:10]}")

    # 6. IDs bruts
    ids_raw = [l.strip() for l in lines_raw if l.strip()]
    submitted_ids = ids_raw

    # 7. Doublons
    seen = set()
    duplicates = []
    for uid in submitted_ids:
        if uid in seen:
            duplicates.append(uid)
        seen.add(uid)
    if duplicates:
        issues.append(f"❌ Doublons ({len(duplicates)}) : {duplicates[:5]}")

    # 8. IDs hors dataset
    dataset_ids = load_user_ids(dataset_json)
    out_of_dataset = [uid for uid in submitted_ids if uid not in dataset_ids]
    if out_of_dataset:
        issues.append(f"❌ CRITIQUE : {len(out_of_dataset)} IDs hors dataset : {out_of_dataset[:5]}")

    # 9. IDs vides ou non-string
    invalid_ids = [uid for uid in submitted_ids if not uid or len(uid) < 2]
    if invalid_ids:
        issues.append(f"❌ IDs invalides (vides ou trop courts) : {invalid_ids[:5]}")

    # 10. Ratio bots/total
    n_total   = len(dataset_ids)
    n_bots    = len(set(submitted_ids))
    bot_ratio = n_bots / n_total if n_total > 0 else 0
    if bot_ratio < 0.05:
        warnings_list.append(f"⚠  Ratio bots très bas : {bot_ratio:.1%} ({n_bots}/{n_total})")
    elif bot_ratio > 0.60:
        warnings_list.append(f"⚠  Ratio bots très élevé : {bot_ratio:.1%} ({n_bots}/{n_total})")

    # 11. Caractères spéciaux (CRLF vs LF)
    if b"\r\n" in raw:
        warnings_list.append("⚠  Fins de ligne Windows (CRLF) — acceptable mais surveiller")

    stats = {
        "file":         str(path),
        "n_submitted":  len(submitted_ids),
        "n_unique":     len(set(submitted_ids)),
        "n_dataset":    n_total,
        "bot_ratio":    round(bot_ratio, 4),
        "size_bytes":   path.stat().st_size,
    }

    ok = len([i for i in issues if i.startswith("❌")]) == 0
    return ok, issues + warnings_list, stats


def main():
    parser = argparse.ArgumentParser(description="Vérification soumission finale BotOrNot")
    parser.add_argument("--en",   required=True, help="Chemin JSON dataset EN")
    parser.add_argument("--fr",   required=True, help="Chemin JSON dataset FR")
    parser.add_argument("--team", required=True, help="Nom de l'équipe")
    parser.add_argument("--out-dir", default="submissions", help="Répertoire de sortie")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    en_file = str(out_dir / f"{args.team}.detections.en.txt")
    fr_file = str(out_dir / f"{args.team}.detections.fr.txt")

    print(f"\n{'='*60}")
    print(f"  🔍 Vérification Finale — {args.team}")
    print(f"{'='*60}")
    print(f"  EN : {en_file}")
    print(f"  FR : {fr_file}")

    all_ok = True

    for lang, file_path, dataset_json in [
        ("EN", en_file, args.en),
        ("FR", fr_file, args.fr),
    ]:
        print(f"\n  {'─'*54}")
        print(f"  Vérification {lang} :")
        ok, issues, stats = check_file(file_path, dataset_json, lang)

        if stats:
            print(f"    IDs soumis      : {stats.get('n_submitted', '?')}")
            print(f"    IDs uniques     : {stats.get('n_unique', '?')}")
            print(f"    Comptes dataset : {stats.get('n_dataset', '?')}")
            print(f"    Ratio bots      : {stats.get('bot_ratio', 0)*100:.1f}%")
            print(f"    Taille fichier  : {stats.get('size_bytes', '?')} octets")

        if not issues:
            print(f"    ✅ Aucun problème détecté")
        else:
            for issue in issues:
                print(f"    {issue}")
                if issue.startswith("❌"):
                    all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print(f"  ✅ VÉRIFICATION RÉUSSIE — Fichiers prêts pour soumission")
    else:
        print(f"  ❌ VÉRIFICATION ÉCHOUÉE — Corriger les erreurs avant envoi")
    print(f"{'='*60}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
