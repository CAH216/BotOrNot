#!/usr/bin/env python
"""
scripts/top_cases_report.py
===========================
Générateur du rapport des "Top Cas" (Mission 11).
Extrait les cas extrêmes (victoires nettes, failles persistantes) 
depuis les prédictions Out-Of-Fold pour aider au diagnostic final.

Exporte :
 - Top 20 Bots évidents
 - Top 20 Humains sauvés par l'Anti-FP
 - Top 20 Faux Positifs tenaces
 - Top 20 Faux Négatifs aveugles
"""
import sys, os, time, json, argparse, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_baseline import _load_file, _find_col, ID_PATTERNS, LABEL_PATTERNS
from optimize_profiles import generate_oofs
from src.inference.anti_fp import AntiFPFilter, AntiFPConfig

SEP = "=" * 70

def _banner(m): print(f"\n{SEP}\n  {m}\n{SEP}")
def _log(m):    print(f"  [{datetime.now():%H:%M:%S}] {m}")


def build_report(args):
    t0 = time.time()
    _banner("MISSION 11 — DIAGNOSTIC DES TOP CAS")

    df_train = _load_file(args.train)
    id_col = _find_col(df_train, ID_PATTERNS) or "user_id"
    label_col = _find_col(df_train, LABEL_PATTERNS)
    if not label_col:
        sys.exit("❌ Colonne de label introuvable.")

    # 1. Obtenir les probas OOF
    feat_train, y_true, account_order, oofs, block_probas = generate_oofs(
        df_train, id_col, label_col, args.cv_folds, args.seed
    )
    
    # Utilisons la moyenne LGBM + CatBoost si disponible, sinon LGBM
    if "lgbm" in oofs and "catboost" in oofs:
        raw_proba = (oofs["lgbm"] + oofs["catboost"]) / 2.0
    elif "lgbm" in oofs:
        raw_proba = oofs["lgbm"]
    else:
        raw_proba = list(oofs.values())[0]

    # 2. Appliquer l'Anti-FP (profil balanced : min 1 mod, pénalité modérée)
    feature_df = feat_train.drop(columns=[id_col]) if id_col in feat_train.columns else feat_train
    base_df = pd.DataFrame({id_col: account_order, "proba": raw_proba})
    
    cfg = AntiFPConfig(
        enabled=True,
        min_modules_for_bot=1,
        unilateral_penalty=0.08,
        pu_min_followers=10000,
        pu_follower_penalty=0.04,
        pu_verified_penalty=0.05,
        conflict_rules_enabled=True
    )
    af_filter = AntiFPFilter(cfg)
    adj_df = af_filter.apply(base_df, feature_df, block_probas)
    
    adj_proba = adj_df["proba_adjusted"].values
    triggered = adj_df["anti_fp_triggered"].values
    
    thresh = args.threshold

    # 3. Consolider les insights modulaires par compte
    # block_probas est un dict {module_name: array_of_probas}
    # np.where sur block probas pour obtenir les "push_to_bot" (> 0.5)
    _log("Analyse des contributions par module...")
    
    records = []
    for i, acc_id in enumerate(account_order):
        mods_bot = []
        mods_hum = []
        for mod, arr in block_probas.items():
            if arr[i] >= 0.5: mods_bot.append(f"{mod}({arr[i]:.2f})")
            else:             mods_hum.append(f"{mod}({arr[i]:.2f})")
            
        records.append({
            "account_id": str(acc_id),
            "true_label": int(y_true[i]),
            "raw_proba": float(raw_proba[i]),
            "adj_proba": float(adj_proba[i]),
            "anti_fp_triggered": bool(triggered[i]),
            "modules_push_bot": ", ".join(mods_bot) if mods_bot else "Aucun",
            "modules_push_hum": ", ".join(mods_hum) if mods_hum else "Aucun",
            "delta_anti_fp": float(raw_proba[i] - adj_proba[i])
        })
        
    df_res = pd.DataFrame(records)

    # 4. Extraction des 4 catégories (Top 20)
    _log("Extraction des Top 20 par catégorie...")
    
    # a. Top 20 Bots évidents (TP)
    df_tp = df_res[(df_res["true_label"] == 1)].sort_values(by="adj_proba", ascending=False).head(20)
    
    # b. Top 20 Humains protégés (TN sauvés)
    # Vrais humains dont la proba brute diminuée par l'Anti-FP
    df_tn_saved = df_res[(df_res["true_label"] == 0) & (df_res["anti_fp_triggered"] == True)]
    df_tn_saved = df_tn_saved.sort_values(by="delta_anti_fp", ascending=False).head(20)
    
    # c. Top 20 Faux Positifs persistants
    df_fp = df_res[(df_res["true_label"] == 0) & (df_res["adj_proba"] >= thresh)]
    df_fp = df_fp.sort_values(by="adj_proba", ascending=False).head(20)
    
    # d. Top 20 Faux Négatifs persistants
    df_fn = df_res[(df_res["true_label"] == 1) & (df_res["adj_proba"] < thresh)]
    df_fn = df_fn.sort_values(by="adj_proba", ascending=True).head(20)

    # 5. Création Markdown complet
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_md = out_dir / "top_cases_report.md"
    target_json = out_dir / "top_cases_report.json"
    
    def _format_md_section(title, df_sub, desc):
        lines = [f"## {title}", f"*{desc}*\n"]
        if df_sub.empty:
            lines.append("Aucun cas dans cette catégorie.\n")
            return "\n".join(lines)
            
        header = "| Account ID | Raw Prob | Adj Prob | Anti-FP | Modules → BOT | Modules → HUMAIN |"
        sep    = "|---|---|---|---|---|---|"
        lines.extend([header, sep])
        for _, r in df_sub.iterrows():
            afp_icon = "🛡️ Oui" if r["anti_fp_triggered"] else "Non"
            row = (f"| `{r['account_id']}` "
                   f"| {r['raw_proba']:.3f} | **{r['adj_proba']:.3f}** "
                   f"| {afp_icon} "
                   f"| {r['modules_push_bot']} | {r['modules_push_hum']} |")
            lines.append(row)
        lines.append("\n")
        return "\n".join(lines)

    md_content = [
        f"# Rapport de Diagnostic — Top Cas (Mission 11)",
        f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Seuil de Décision** : {thresh}",
        f"**Filtre Anti-FP** : `balanced`\n"
    ]
    
    md_content.append(_format_md_section(
        "1. Les 20 Bots les plus évidents (Vrais Positifs Forts)", df_tp, 
        "Les comptes bots que le modèle détecte avec la plus forte certitude."
    ))
    
    md_content.append(_format_md_section(
        "2. Les 20 Humains héroïquement sauvés (True Negatives protégés)", df_tn_saved, 
        "Les comptes humains que le modèle trouvait suspects (proba proche ou > seuil) mais que le filtre Anti-Faux-Positifs a pénalisés à juste titre."
    ))
    
    md_content.append(_format_md_section(
        "3. Les 20 Faux Positifs tenaces", df_fp, 
        "Les humains que le pipeline continue de bannir (Erreur critique). Observez les modules qui trompent le score."
    ))
    
    md_content.append(_format_md_section(
        "4. Les 20 Faux Négatifs indétectables", df_fn, 
        "Les bots qui passent totalement sous le radar (vus comme très humains). Observez l'absence de signaux."
    ))
    
    with open(target_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    # JSON Export
    export_dict = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "threshold": thresh,
            "description": "Top cases extracted via CV OOF predictions."
        },
        "top_tp": df_tp.to_dict(orient="records"),
        "top_tn_saved": df_tn_saved.to_dict(orient="records"),
        "top_fp": df_fp.to_dict(orient="records"),
        "top_fn": df_fn.to_dict(orient="records")
    }
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(export_dict, f, indent=2, ensure_ascii=False)

    _banner(f"✅ Rapport généré ({time.time() - t0:.1f}s)")
    _log(f"Export Markdown : {target_md}")
    _log(f"Export JSON     : {target_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.50, help="Seuil de classification.")
    p.add_argument("--out", default="artifacts/top_cases")
    args = p.parse_args()
    build_report(args)
