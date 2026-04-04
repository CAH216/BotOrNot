# -*- coding: utf-8 -*-
"""
src/features/candidate_miner_court.py
=======================================
Residual Candidate Miner + Pairwise Court.

Architecture :
  1. CandidateMiner  — Nomination conservative + Veto (tous les Bots) + Expansion (E30)
  2. PairwiseCourt   — Vote multi-espace (5 espaces, comparaison NN)
  3. Garde-fous      — Override conditionnel (Veto vs Rescue)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Espaces de features ────────────────────────────────────────────────────────

SPACE_PREFIXES = {
    "Forensic":  ["forensic_"],
    "Temporal":  ["tmp_", "clock_"],
    "Text":      ["txt_", "gh_"],
    "UserMeta":  ["usr_", "contra_"],
    "Struct":    ["vas_", "lrh_", "rel_"],
}

def _get_space_cols(all_cols, space_name):
    prefixes = SPACE_PREFIXES.get(space_name, [])
    return [c for c in all_cols if any(c.startswith(p) for p in prefixes)]

# ── Candidate Miner ────────────────────────────────────────────────────────────

class CandidateMiner:
    """
    Nomme les comptes qui méritent un jugement du Court.
    
    Path 1 (Veto) : Tous les comptes avec prob >= 0.5 (si use_veto=True)
    Path 2 (Standard Rescue) : 0.01 < prob < prob_high ET forensic > P_event ET lrh2 < cap
    Path 3 (Expansion Rescue) : 0.01 < prob < prob_high ET lrh2 < cap ET features temporelles fortes
    """
    def __init__(
        self,
        proba_low  = 0.01,
        proba_high = 0.35,
        forensic_percentile = 65,   
        human_archetype_cap = 0.30, 
        use_expansion       = False,
        use_veto            = False,
    ):
        self.proba_low           = proba_low
        self.proba_high          = proba_high
        self.forensic_percentile = forensic_percentile
        self.human_cap           = human_archetype_cap
        self.use_expansion       = use_expansion
        self.use_veto            = use_veto

    def nominate(
        self,
        uids:            list,
        probs:           np.ndarray,
        feat_df:         pd.DataFrame,
        forensic_scores: pd.Series,
        lrh2_scores:     pd.Series,
    ) -> list:
        event_p = float(np.nanpercentile(forensic_scores.values, self.forensic_percentile))
        candidates = []

        for uid, prob in zip(uids, probs):
            # 1. Path Veto : on veut inspecter les futures "condamnations" du modèle
            if self.use_veto and prob >= 0.5:
                poll_sc = float(feat_df.at[uid, "lrh_poll_score"]) if "lrh_poll_score" in feat_df.columns else 0
                lrh2    = float(lrh2_scores.get(uid, 0))
                # On ne nomine pour un Veto que si l'archétype humain est plausible (poll, stan, rp)
                if poll_sc > 1.0 or lrh2 > 0.2:
                    candidates.append(uid)
                continue

            # 2 & 3 concernent uniquement la fenêtre de sauvetage (Rescue)
            if not (self.proba_low < prob < self.proba_high):
                continue

            # Path Expansion (E30) - Bypasser la bio et lrh2 si signature forte (Temporel + Poll)
            if self.use_expansion and uid in feat_df.index:
                tmp_cv = float(feat_df.at[uid, "tmp_ipt_cv"]) if "tmp_ipt_cv" in feat_df.columns else 0
                poll_sc = float(feat_df.at[uid, "lrh_poll_score"]) if "lrh_poll_score" in feat_df.columns else 0
                
                # La cible E30 affiche une combinaison d'instabilité temporelle + signal poll modéré
                if tmp_cv > 1.5 and poll_sc > 0.05:
                    candidates.append(uid)
                    continue

            lrh2 = float(lrh2_scores.get(uid, 0))
            if lrh2 >= self.human_cap:
                continue

            # Path Standard Rescue
            fscore = float(forensic_scores.get(uid, 0))
            if fscore > event_p:
                candidates.append(uid)

        return list(set(candidates))

# ── Pairwise Court ─────────────────────────────────────────────────────────────

class PairwiseCourt:
    def __init__(self, k=3, min_bot_votes=2):
        self.k             = k
        self.min_bot_votes = min_bot_votes
        # ── Banque de Référence Gelée (remplitée dans fit()) ─────────────────
        self._fitted    = False
        self._scalers   = {}   # space_name -> StandardScaler
        self._bot_vecs  = {}   # space_name -> np.ndarray (n_bots x d)
        self._hum_vecs  = {}   # space_name -> np.ndarray (n_hums x d)
        self._ref_cols  = {}   # space_name -> list[str]

    # ── Fit (appelé une fois par fold, sur le TRAINING SET uniquement) ──
    def fit(self, ref_df: pd.DataFrame, ref_labels: pd.Series) -> "PairwiseCourt":
        """
        Construit la banque de référence figée depuis le fold d'entraînement.
        
        ref_df     : Features du training set (index = user_id)
        ref_labels : Série {user_id -> 0/1} pour le training set
        """
        all_cols = ref_df.columns.tolist()
        uids = list(ref_df.index)
        labels_arr = ref_labels.loc[uids].values

        bot_idx = np.where(labels_arr == 1)[0]
        hum_idx = np.where(labels_arr == 0)[0]

        for space_name in SPACE_PREFIXES:
            scols = _get_space_cols(all_cols, space_name)
            valid = [c for c in scols if c in ref_df.columns]
            if len(valid) < 2:
                continue

            X_sub = ref_df[valid].fillna(0).values.astype(float)
            sc = StandardScaler()
            X_sc = sc.fit_transform(X_sub)

            if len(bot_idx) == 0 or len(hum_idx) == 0:
                continue

            self._scalers[space_name]  = sc
            self._ref_cols[space_name] = valid
            self._bot_vecs[space_name] = X_sc[bot_idx]   # Vecteurs BOTS figés
            self._hum_vecs[space_name] = X_sc[hum_idx]   # Vecteurs HUMAINS figés

        self._fitted = True
        return self

    # ── Adjudicate (invariant : ref bank fixée, seul le candidat change) ──
    def adjudicate(
        self,
        candidate_uid: str,
        feat_df:        pd.DataFrame,   # contient LE candidat
        labels:         pd.Series,      # non utilisé si fitted (compatibilité amont)
        all_cols:       list,
    ) -> dict:
        # ─ Mode Invariant (Banque Figée présente) ────────────────────
        if self._fitted:
            return self._adjudicate_invariant(candidate_uid, feat_df)
        # ─ Mode Legacy (compatibilité arrière si pas de fit()) ────────
        return self._adjudicate_legacy(candidate_uid, feat_df, labels, all_cols)

    def _adjudicate_invariant(self, candidate_uid: str, feat_df: pd.DataFrame) -> dict:
        """
        Mode invariant amélioré :
        - Les identités de référence (quels comptes sont bots/humains) viennent du training.
        - Les distances sont calculées dans l'espace du TEST EVENT (scaler local).

        Mécanisme :
          1. Inverser le scaler training -> vecteurs bruts des références.
          2. Fitter un scaler local sur le test event.
          3. Projeter références + candidat dans cet espace local.
          4. KNN dans l'espace local calibré.
        """
        votes = {}
        for space_name in self._scalers:
            valid           = self._ref_cols[space_name]
            sc_tr           = self._scalers[space_name]
            bot_vecs_scaled = self._bot_vecs[space_name]
            hum_vecs_scaled = self._hum_vecs[space_name]

            row_cols = [c for c in valid if c in feat_df.columns]
            if len(row_cols) < 2:
                continue

            # 1. Inverser scaler training -> vecteurs bruts
            bot_vecs_raw = sc_tr.inverse_transform(bot_vecs_scaled)
            hum_vecs_raw = sc_tr.inverse_transform(hum_vecs_scaled)

            # 2. Aligner le test event sur la liste 'valid'
            X_test_aligned = np.zeros((len(feat_df), len(valid)))
            for j, c in enumerate(valid):
                if c in feat_df.columns:
                    X_test_aligned[:, j] = feat_df[c].fillna(0).values.astype(float)

            # 3. Scaler local (distribution du test event)
            sc_local = StandardScaler()
            sc_local.fit(X_test_aligned)

            bot_vecs_local = sc_local.transform(bot_vecs_raw)
            hum_vecs_local = sc_local.transform(hum_vecs_raw)

            raw = np.zeros((1, len(valid)))
            for j, c in enumerate(valid):
                if c in feat_df.columns and candidate_uid in feat_df.index:
                    raw[0, j] = float(feat_df.at[candidate_uid, c])
            vec_s = sc_local.transform(raw)[0]

            # 4. Distances KNN
            d_bots = np.linalg.norm(bot_vecs_local - vec_s, axis=1)
            d_hums = np.linalg.norm(hum_vecs_local - vec_s, axis=1)

            top_bot = float(np.sort(d_bots)[:self.k].mean()) if len(d_bots) >= self.k else float(d_bots.mean())
            top_hum = float(np.sort(d_hums)[:self.k].mean()) if len(d_hums) >= self.k else float(d_hums.mean())
            margin  = top_hum - top_bot

            votes[space_name] = {
                "vote":   "bot" if top_bot < top_hum else "human",
                "d_bot":  round(top_bot, 3),
                "d_hum":  round(top_hum, 3),
                "margin": round(margin, 3),
            }

        bot_votes  = sum(1 for v in votes.values() if v["vote"] == "bot")
        hum_votes  = sum(1 for v in votes.values() if v["vote"] == "human")
        avg_margin = float(np.mean([v["margin"] for v in votes.values()])) if votes else 0.0

        return {
            "spaces": votes, "bot_votes": bot_votes,
            "human_votes": hum_votes, "avg_margin": round(avg_margin, 3)
        }

    def _adjudicate_legacy(
        self,
        candidate_uid: str,
        feat_df:        pd.DataFrame,
        labels:         pd.Series,
        all_cols:       list,
    ) -> dict:
        """Mode legacy : KNN sur tout feat_df (comportement original). Conservé pour compatibilité."""
        uids_list = list(feat_df.index)
        if candidate_uid not in uids_list:
            return {"bot_votes": 0, "human_votes": len(SPACE_PREFIXES)}

        i_cand = uids_list.index(candidate_uid)
        labels_arr = labels.loc[uids_list].values

        bot_idx = np.where(labels_arr == 1)[0]
        hum_idx = np.where(labels_arr == 0)[0]

        votes = {}
        for space_name in SPACE_PREFIXES:
            scols = _get_space_cols(all_cols, space_name)
            valid = [c for c in scols if c in feat_df.columns]
            if len(valid) < 2:
                continue

            X    = feat_df[valid].fillna(0).values.astype(float)
            X_sc = StandardScaler().fit_transform(X)
            vec  = X_sc[i_cand].reshape(1, -1)

            bot_idx_clean = [j for j in bot_idx if j != i_cand]
            hum_idx_clean = [j for j in hum_idx if j != i_cand]
            if not bot_idx_clean or not hum_idx_clean:
                continue

            d_bots = np.linalg.norm(X_sc[bot_idx_clean] - vec, axis=1)
            d_hums = np.linalg.norm(X_sc[hum_idx_clean] - vec, axis=1)

            top_bot = float(np.sort(d_bots)[:self.k].mean())
            top_hum = float(np.sort(d_hums)[:self.k].mean())
            margin  = top_hum - top_bot

            votes[space_name] = {
                "vote":   "bot" if top_bot < top_hum else "human",
                "d_bot":  round(top_bot, 3),
                "d_hum":  round(top_hum, 3),
                "margin": round(margin, 3),
            }

        bot_votes  = sum(1 for v in votes.values() if v["vote"] == "bot")
        hum_votes  = sum(1 for v in votes.values() if v["vote"] == "human")
        avg_margin = float(np.mean([v["margin"] for v in votes.values()])) if votes else 0.0

        return {
            "spaces": votes, "bot_votes": bot_votes,
            "human_votes": hum_votes, "avg_margin": round(avg_margin, 3)
        }

class AtypicalHumanRescueCourt:
    """
    Tribunal spécialisé K-NN constitué EXCLUSIVEMENT des vrais humains atypiques du set d'entrainement.
    Permet de contourner la densité collégiale bot.
    """
    def __init__(self, train_feat: pd.DataFrame, train_labels: pd.Series):
        if "forensic_bot_score" in train_feat.columns:
            # On isole les humains flaggés toxiques par le modèle
            mask = (train_labels == 0) & (train_feat["forensic_bot_score"] >= 0.45)
        else:
            # Fallback
            mask = (train_labels == 0)
            
        self.bank_feat = train_feat[mask]
        self.scaler = None
        self.knn = None
        
    def build(self, feature_cols: list):
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        if self.bank_feat.empty: return
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(self.bank_feat[feature_cols].fillna(0))
        self.knn = NearestNeighbors(n_neighbors=3, metric="cosine")
        self.knn.fit(X_s)
        
    def find_rescue_neighbors(self, u_feat: pd.Series, feature_cols: list) -> int:
        """ Renvoie le nombre de voisins atypiques extrêmement proches. """
        if self.knn is None: return 0
        x_s = self.scaler.transform(pd.DataFrame([u_feat[feature_cols].fillna(0)]))
        dists, _ = self.knn.kneighbors(x_s)
        # Seuil serré (cosine dist très proche < 0.15)
        close_neighbors = sum(1 for d in dists[0] if d <= 0.15)
        return close_neighbors

# ── Pipeline Intégré (Court + Miner) ───────────────────────────────────────────

def run_appeal_pipeline(
    uids:            list,
    probs_base:      np.ndarray,
    feat_df:         pd.DataFrame,
    forensic_df:     pd.DataFrame,
    labels:          pd.Series,
    all_cols:        list,
    miner: CandidateMiner  = None,
    court: PairwiseCourt   = None,
    verbose: bool          = False,
    posts_df:        pd.DataFrame = None,
    train_feat:      pd.DataFrame = None,
    train_labels:    pd.Series = None,
    arbitration_mode: str = "PASSIVE"
) -> tuple:
    if miner is None: miner = CandidateMiner()
    if court is None: court = PairwiseCourt()

    preds = (probs_base >= 0.5).astype(int).copy()

    f_col = "forensic_bot_score"
    l_col = "lrh2_residual_score"
    forensic_scores = forensic_df[f_col] if f_col in forensic_df.columns else pd.Series(0, index=feat_df.index)
    lrh2_scores     = feat_df[l_col]     if l_col in feat_df.columns     else pd.Series(0, index=feat_df.index)

    forens_cols = [c for c in forensic_df.columns if c.startswith("forensic_")]
    feat_aug = feat_df.copy()
    for c in forens_cols:
        if c in forensic_df.columns:
            feat_aug[c] = forensic_df[c]

    # Load Unified Groq General Judge if posts data is available
    from src.features.groq_general_judge import GroqGeneralJudge
    from src.features.final_arbitration_judge import FinalArbitratorJudge
    from src.features.groq_forced_judge import GroqForcedJudge
    
    general_judge = None
    forced_judge = None
    if posts_df is not None and train_feat is not None and train_labels is not None:
        general_judge = GroqGeneralJudge(u_df=feat_df, p_df=posts_df, train_feat=train_feat, train_labels=train_labels)
        forced_judge = GroqForcedJudge(u_df=feat_df, p_df=posts_df, train_feat=train_feat, train_labels=train_labels)

    rescue_court = None
    if "DUAL_COURT" in arbitration_mode and train_feat is not None and train_labels is not None:
        rescue_court = AtypicalHumanRescueCourt(train_feat, train_labels)
        rescue_court.build(all_cols)

    candidates = miner.nominate(uids, probs_base, feat_df, forensic_scores, lrh2_scores)

    appeal_log = {}
    for uid in candidates:
        if uid not in feat_aug.index: continue
        
        idx = list(uids).index(uid)
        prob = probs_base[idx]
        is_bot_pred = prob >= 0.5
        
        result = court.adjudicate(uid, feat_aug, labels, list(feat_aug.columns))
        bot_votes = result["bot_votes"]
        hum_votes = result.get("human_votes", 0)
        margin = result.get("avg_margin", 0.0)
        
        action = "none"
        
        # 1. Action Supreme: (DUAL COURT MICRO-VETO)
        if rescue_court is not None and prob >= 0.50:
            rescue_hits = rescue_court.find_rescue_neighbors(feat_aug.loc[uid], all_cols)
            if rescue_hits >= 1:
                # LLM CALL IF DUAL_COURT_ACTIVE, ELSE PASSIVE VETO
                if arbitration_mode == "DUAL_COURT_ACTIVE" and forced_judge is not None:
                    try:
                        res = forced_judge.evaluate_fp_veto(uid, prob, result["spaces"], feat_aug)
                        if res.get("verdict") == "HUMAN" and res.get("confidence", 0) >= 0.80:
                            preds[idx] = 0
                            override = "dual_court_llm_rescued"
                            action = "llm_veto"
                            result["action"] = action
                            appeal_log[uid] = {"prob": prob, "override": override, "reason": "Dual Court Match + LLM"}
                            continue
                    except:
                        pass
                
                elif arbitration_mode == "DUAL_COURT_PASSIVE":
                    preds[idx] = 0
                    override = "dual_court_passive_rescued"
                    action = "veto"
                    result["action"] = action
                    appeal_log[uid] = {"prob": prob, "override": override, "reason": "Dual Court Match"}
                    continue
        
        # 2. Action Veto (Protéger l'humain accusé Historique)
        if is_bot_pred and miner.use_veto and action == "none":
            # ── GARDE-FOU prob extrême ─────────────────────────────────────────
            # Un compte avec prob >= 0.97 ne peut PAS être vetoé sans consensus
            # humain TRÈS fort du Court (0 vote bot ET >= 3 votes humains).
            # Raison : si le modèle principal est quasi-certain, le Court
            # (qui opère dans un espace de features différent) ne doit pas
            # annuler la décision sur un simple manque de voisins bots.
            # Coût du bug corrigé : @orealways3 prob=1.000, @KatieXO prob=0.976
            # vetoed avec 0 votes → 2 FN = -4 pts.
            if prob >= 0.97:
                # Veto autorisé seulement si Court TRÈS massivement humain
                veto_ok = (bot_votes == 0 and hum_votes >= 3)
            else:
                # Comportement normal : veto si bot_votes <= 1
                veto_ok = (bot_votes <= 1)

            if veto_ok:
                preds[idx] = 0
                action = "veto"

            # 3. Action Rescue standard
        else:
            if action == "none":
                action = "rescue" if margin <= -0.5 and bot_votes < 2.5 else "none"
                if action == "rescue":
                    preds[idx] = 0
                    override = "court_rescued"
                    
        # 4. Action Supreme: Groq General / Forced LLM
        if general_judge is not None and forced_judge is not None:
            # ---> MODES MICRO-VETO FP ONLY
            if arbitration_mode in ["MICRO_V1", "MICRO_V2", "MICRO_V3"] and prob >= 0.50:
                req_h_trig = 2 if arbitration_mode == "MICRO_V2" else 1
                req_c = 0.85 if arbitration_mode == "MICRO_V3" else 0.80
                
                if hum_votes >= req_h_trig:
                    try:
                        res = forced_judge.evaluate_fp_veto(uid, prob, result["spaces"], feat_aug)
                        v = res.get("verdict", "")
                        c = res.get("confidence", 0.0)
                        m = res.get("margin", 0.0)
                        if v == "HUMAN" and c >= req_c and m >= 0.30:
                            preds[idx] = 0
                            override = f"forced_{arbitration_mode}_veto"
                            action = "llm_human"
                    except:
                        pass
                        
            # ---> MODE FP_RESCUE
            elif arbitration_mode in ["FP_RESCUE_JUDGE", "COMBINED"] and prob >= 0.50 and hum_votes >= 3:
                try:
                    res = forced_judge.evaluate_fp_veto(uid, prob, result["spaces"], feat_aug)
                    v = res.get("verdict", "")
                    c = res.get("confidence", 0.0)
                    m = res.get("margin", 0.0)
                    if v == "HUMAN" and c >= 0.90 and m >= 0.20:
                        preds[idx] = 0
                        override = "forced_fp_veto"
                        action = "llm_human"
                except:
                    pass

            # ---> MODE FN_RESCUE
            elif arbitration_mode in ["FN_RESCUE_JUDGE", "COMBINED"] and prob <= 0.49 and bot_votes >= 2:
                try:
                    res = forced_judge.evaluate_fn_rescue(uid, prob, result["spaces"], feat_aug)
                    v = res.get("verdict", "")
                    c = res.get("confidence", 0.0)
                    m = res.get("margin", 0.0)
                    if v == "BOT" and c >= 0.92 and m >= 0.20:
                        preds[idx] = 1
                        override = "forced_fn_rescue"
                        action = "llm_bot"
                except:
                    pass

            # ---> MODE ANCIEN PASSif
            elif arbitration_mode == "PASSIVE":
                trigger_llm = False
                if prob >= 0.60 and hum_votes >= 3: trigger_llm = True
                if prob <= 0.49 and bot_votes >= 2: trigger_llm = True
                if trigger_llm:
                    try:
                        is_fr = any(c.startswith("fr_") for c in feat_df.columns)
                        mode = "FR_CLEAN_PERSONA" if is_fr else "EN_OVERPLAYED_HUMAN"
                        nomination_reason = "Passive Target"
                        res = general_judge.evaluate_account(uid, float(prob), result["spaces"], nomination_reason, feat_aug, judge_mode=mode)
                        arb = FinalArbitratorJudge.arbitrate(prob, result["spaces"], res, arbitration_mode)
                        final_action = arb["final_action"]
                        if final_action == "KEEP_HUMAN":
                            preds[idx] = 0
                            override = "groq_passive_veto"
                            action = "llm_human"
                        elif final_action == "KEEP_BOT":
                            preds[idx] = 1
                            override = "groq_passive_rescue"
                            action = "llm_bot"
                    except:
                        pass

        result["action"] = action
        result["prob"] = prob
        appeal_log[uid] = result

    # ── Hook de Conviction par Détecteur LLM ──────────────────────────────────
    llm_cols = [
        "llm_control_chars_count",
    ]
    if "llm_control_chars_count" in feat_df.columns:
        for idx, uid in enumerate(uids):
            # Caractères de contrôle (\u0004, \b, mojibake sévère)
            if feat_df.loc[uid, "llm_control_chars_count"] > 0 and preds[idx] == 0:
                preds[idx] = 1
                if uid not in appeal_log:
                    appeal_log[uid] = {"prob": probs_base[idx]}
                appeal_log[uid]["action"] = "hard_convict_llm_mojibake"

    return preds, appeal_log
