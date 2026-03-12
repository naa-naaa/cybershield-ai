import asyncio
import numpy as np
from typing import List, Optional

from agents.text.agent_t1_toxicity import AgentT1Toxicity
from agents.text.agent_t2_context import AgentT2Context
from agents.text.agent_t3_emotion import AgentT3Emotion
from preprocessing.preprocessor import TextPreprocessor


class TextOrchestrator:
    """
    Text Orchestrator.
    Runs preprocessing → dispatches T1/T2/T3 in parallel → fuses scores.
    """

    def __init__(self):
        print("[Orchestrator] Initializing text agents...")
        self.preprocessor = TextPreprocessor()
        self.t1 = AgentT1Toxicity()
        self.t2 = AgentT2Context()
        self.t3 = AgentT3Emotion()
        print("[Orchestrator] All text agents ready")

        self.W_T1 = 0.50
        self.W_T2 = 0.25
        self.W_T3 = 0.25
        self.THRESHOLD_ALLOW = 0.40
        self.THRESHOLD_BLOCK = 0.80

    def _fuse_scores(self, t1: dict, t2: dict, t3: dict) -> float:
        base = self.W_T1 * t1["score"] + self.W_T2 * t2["score"] + self.W_T3 * t3["score"]
        if t2.get("escalation_flag"):
            base += 0.08
        if t2.get("repetition_flag"):
            base += 0.05
        if t3.get("victim_distress_flag"):
            base += 0.05
        if t1.get("women_risk_flag"):
            base += 0.07
        return float(np.clip(base, 0.0, 1.0))

    def _determine_action(self, score: float, strike_count: int) -> str:
        if strike_count >= 3:
            return "block"
        if score < self.THRESHOLD_ALLOW:
            return "allow"
        elif score < self.THRESHOLD_BLOCK:
            return "alert"
        return "block"

    def _build_explanation(self, t1, t2, t3, score) -> str:
        parts = []
        if t1["score"] > 0.5:
            parts.append(f"{t1['category']} detected ({t1['score']:.0%})")
        if t2.get("escalation_flag"):
            parts.append("escalating thread pattern")
        if t2.get("repetition_flag"):
            parts.append(f"repeated {t2.get('repeat_count',0)} times")
        if t3.get("dominant_emotion") in ["anger", "disgust", "fear"]:
            parts.append(f"aggressive tone: {t3['dominant_emotion']}")
        if t3.get("victim_distress_flag"):
            parts.append("victim distress signals detected")
        if t1.get("women_risk_flag"):
            parts.append("gender-targeted content detected")
        return "; ".join(parts) if parts else "Message analyzed"

    async def analyze(
        self,
        text: str,
        thread_history: Optional[List[str]] = None,
        sender: str = "",
        strike_count: int = 0,
    ) -> dict:
        thread_history = thread_history or []

        # Preprocessing: handles Tamil, Tanglish, emojis
        pre_result = self.preprocessor.process(text)
        clean_text = pre_result["processed_text"]
        preprocessing_boost = pre_result.get("preprocessing_boost", 0.0)

        # Dispatch all three agents simultaneously
        t1_result, t2_result, t3_result = await asyncio.gather(
            self.t1.analyze(clean_text),
            self.t2.analyze(clean_text, thread_history),
            self.t3.analyze(clean_text),
        )

        fused_score = self._fuse_scores(t1_result, t2_result, t3_result)
        fused_score = float(np.clip(fused_score + preprocessing_boost, 0.0, 1.0))

        action = self._determine_action(fused_score, strike_count)
        explanation = self._build_explanation(t1_result, t2_result, t3_result, fused_score)

        category = t1_result.get("category", "safe")
        if fused_score < self.THRESHOLD_ALLOW:
            category = "safe"

        if fused_score < 0.40:
            severity = "none"
        elif fused_score < 0.65:
            severity = "low"
        elif fused_score < 0.80:
            severity = "medium"
        else:
            severity = "high"

        return {
            "harm_score": fused_score,
            "action": action,
            "category": category,
            "severity": severity,
            "explanation": explanation,
            "women_risk_flag": t1_result.get("women_risk_flag", False),
            "victim_distress_flag": t3_result.get("victim_distress_flag", False),
            "agent_scores": {
                "toxicity": t1_result["score"],
                "context": t2_result["score"],
                "emotion": t3_result["score"],
            },
            "agent_details": {"t1": t1_result, "t2": t2_result, "t3": t3_result},
            "preprocessing": pre_result,
        }
