from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class AgentT2Context:
    """
    Context and Thread Agent.
    Uses Sentence-BERT to detect escalation patterns across threads.
    """

    def __init__(self):
        print("[T2] Loading Sentence-BERT model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.escalation_threshold = 0.35
        print("[T2] Sentence-BERT loaded")

    def _compute_escalation(self, embeddings: np.ndarray) -> float:
        if len(embeddings) < 2:
            return 0.0
        first = embeddings[0]
        last = embeddings[-1]
        similarity = np.dot(first, last) / (
            np.linalg.norm(first) * np.linalg.norm(last) + 1e-8
        )
        return float(np.clip(1.0 - similarity, 0.0, 1.0))

    def _check_repetition(self, messages: List[str]) -> dict:
        if len(messages) < 2:
            return {"repetition_score": 0.0, "repeat_count": 0}
        embeddings = self.model.encode(messages)
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(sim)
        avg_sim = float(np.mean(similarities))
        repetition_score = avg_sim if avg_sim > 0.7 else 0.0
        repeat_count = sum(1 for s in similarities if s > 0.85)
        return {"repetition_score": repetition_score, "repeat_count": repeat_count}

    async def analyze(self, text: str, thread_history: List[str] = []) -> dict:
        try:
            all_messages = thread_history + [text]
            if len(all_messages) < 2:
                return {
                    "agent": "T2_context",
                    "score": 0.0,
                    "escalation_flag": False,
                    "repetition_flag": False,
                    "target_identified": False,
                    "thread_length": len(all_messages),
                    "explanation": "Insufficient thread history",
                }
            embeddings = self.model.encode(all_messages)
            escalation_score = self._compute_escalation(embeddings)
            repetition = self._check_repetition(all_messages)
            target_identified = len(all_messages) >= 5

            context_score = (
                0.50 * escalation_score
                + 0.30 * repetition["repetition_score"]
                + 0.20 * (0.8 if target_identified else 0.0)
            )

            return {
                "agent": "T2_context",
                "score": float(np.clip(context_score, 0.0, 1.0)),
                "escalation_score": escalation_score,
                "escalation_flag": escalation_score > self.escalation_threshold,
                "repetition_flag": repetition["repeat_count"] >= 2,
                "repeat_count": repetition["repeat_count"],
                "target_identified": target_identified,
                "thread_length": len(all_messages),
                "explanation": f"Escalation {escalation_score:.2f}, repeats: {repetition['repeat_count']}",
            }
        except Exception as e:
            print(f"[T2] Analysis error: {e}")
            return {
                "agent": "T2_context",
                "score": 0.0,
                "escalation_flag": False,
                "repetition_flag": False,
                "target_identified": False,
                "thread_length": 0,
                "explanation": "Context analysis failed",
            }
