import torch
from transformers import pipeline
import numpy as np


class AgentT3Emotion:
    """
    Emotion and Tone Analysis Agent.
    Detects anger, fear, disgust, victim distress, sarcasm.
    """

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print("[T3] Loading emotion classifier...")
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=self.device,
        )
        print("[T3] Emotion model loaded")
        self.aggression_emotions = {"anger", "disgust", "fear"}
        self.distress_keywords = [
            "help me", "scared", "afraid", "please stop",
            "leave me alone", "i'm scared", "crying",
            "cant breathe", "i hate myself", "nobody cares",
            "want to disappear",
        ]

    def _compute_aggression_score(self, emotion_scores: dict) -> float:
        aggression = sum(emotion_scores.get(e, 0.0) for e in self.aggression_emotions)
        return float(np.clip(aggression, 0.0, 1.0))

    def _check_victim_distress(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.distress_keywords)

    def _check_sarcasm(self, text: str, dominant_emotion: str) -> bool:
        sarcasm_markers = ["sure", "right", "obviously", "totally", "wow", "great job", "nice one", "brilliant"]
        return any(m in text.lower() for m in sarcasm_markers) and dominant_emotion in {"joy", "neutral"}

    async def analyze(self, text: str) -> dict:
        try:
            results = self.classifier(text[:512])
            emotion_scores = {r["label"].lower(): r["score"] for r in results[0]}
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            aggression_score = self._compute_aggression_score(emotion_scores)
            victim_distress = self._check_victim_distress(text)
            sarcasm_flag = self._check_sarcasm(text, dominant_emotion)

            if sarcasm_flag:
                aggression_score = min(1.0, aggression_score + 0.15)
            if victim_distress:
                aggression_score = min(1.0, aggression_score + 0.10)

            return {
                "agent": "T3_emotion",
                "score": aggression_score,
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores,
                "aggression_score": aggression_score,
                "victim_distress_flag": victim_distress,
                "sarcasm_flag": sarcasm_flag,
                "explanation": f"Dominant: {dominant_emotion}, Aggression: {aggression_score:.2f}",
            }
        except Exception as e:
            print(f"[T3] Analysis error: {e}")
            return {
                "agent": "T3_emotion",
                "score": 0.0,
                "dominant_emotion": "neutral",
                "emotion_scores": {},
                "aggression_score": 0.0,
                "victim_distress_flag": False,
                "sarcasm_flag": False,
                "explanation": "Emotion analysis failed",
            }
