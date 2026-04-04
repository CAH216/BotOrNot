import json
from typing import Dict, Any

class FinalArbitratorJudge:
    """
    Le Meta-Arbitre Python chargé de sécuriser la décision entre les 3 piliers:
    - LightGBM (Probability)
    - K-NN Court (Votes)
    - LLM (Groq General JSON)
    """

    @staticmethod
    def arbitrate(lgbm_prob: float, court_votes: dict, llm_json: dict, arbitration_mode: str = "PASSIVE") -> Dict[str, Any]:
        result = {
            "final_action": "ABSTAIN",
            "action_confidence": 0.0,
            "rationale_short": ""
        }
        
        bot_votes = sum(1 for v in court_votes.values() if v["vote"] == "bot")
        hum_votes = sum(1 for v in court_votes.values() if v["vote"] == "human")
        
        llm_reco = llm_json.get("final_recommendation", "ABSTAIN")
        llm_conf = float(llm_json.get("confidence", 0.0))
        
        # Définition des seuils selon le Mode
        req_h_votes_ext = 2 if arbitration_mode in ["ACTIVE_FP", "ACTIVE_COMBINED"] else 3
        req_h_conf_ext  = 0.85 if arbitration_mode in ["ACTIVE_FP", "ACTIVE_COMBINED"] else 0.95
        
        req_h_votes_std = 1 if arbitration_mode in ["ACTIVE_FP", "ACTIVE_COMBINED"] else 2
        req_h_conf_std  = 0.70 if arbitration_mode in ["ACTIVE_FP", "ACTIVE_COMBINED"] else 0.80
        
        req_b_votes_std = 1 if arbitration_mode in ["ACTIVE_FN", "ACTIVE_COMBINED"] else 2
        req_b_conf_std  = 0.75 if arbitration_mode in ["ACTIVE_FN", "ACTIVE_COMBINED"] else 0.85
        
        # RULE 1 : La Forteresse ML (Certitude Absolue)
        if lgbm_prob >= 0.97:
            # Ne permet l'inversion que sous convergence absolue ou relâchée
            if hum_votes >= req_h_votes_ext and llm_reco == "HUMAN" and llm_conf >= req_h_conf_ext:
                result["final_action"] = "KEEP_HUMAN"
                result["action_confidence"] = llm_conf
                result["rationale_short"] = "VETO PROTECTEUR EXCEPTIONNEL : Le ML est certain à >97%, mais le K-NN Court et l'LLM convergent massivement vers l'Humain (>95%)."
                return result
            else:
                result["final_action"] = "ABSTAIN"
                result["rationale_short"] = "BLOCKED : Prob ML >= 0.97 sans convergence d'acquittement absolue."
                return result

        # RULE 2 : Veto Protecteur Standard (ML modéré à élevé)
        if lgbm_prob >= 0.50:
            if hum_votes >= req_h_votes_std and llm_reco == "HUMAN" and llm_conf >= req_h_conf_std:
                result["final_action"] = "KEEP_HUMAN"
                result["action_confidence"] = llm_conf
                result["rationale_short"] = "VETO STANDARD : Le modèle ML accuse, mais le Court et le LLM s'alignent pour innocenter."
                return result
                
        # RULE 3 : Rescue Punitif (ML faible, Bot Furtif)
        if lgbm_prob <= 0.49:
            if bot_votes >= req_b_votes_std and llm_reco == "BOT" and llm_conf >= req_b_conf_std:
                result["final_action"] = "KEEP_BOT"
                result["action_confidence"] = llm_conf
                result["rationale_short"] = "RESCUE STANDARD : Le modèle ML a raté le Bot, mais Court et LLM convergent vers la culpabilité."
                return result

        # DEFAULT : Prudence
        result["final_action"] = "ABSTAIN"
        result["rationale_short"] = "DEFAULT : Pas de convergence stricte atteinte."
        return result
