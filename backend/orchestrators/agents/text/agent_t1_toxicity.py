import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from detoxify import Detoxify
import numpy as np
import os


class AgentT1Toxicity:
    """
    Toxicity Identification Agent.
    Uses fine-tuned RoBERTa if available, falls back to Detoxify.
    """

    def __init__(self):
        self.model_path = os.path.join(os.getenv("MODEL_PATH", "./models"), "text_toxicity")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.detoxify_model = None
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path) and os.listdir(self.model_path):
                print(f"[T1] Loading fine-tuned model from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path
                ).to(self.device)
                self.model.eval()
                print("[T1] Fine-tuned model loaded")
            else:
                print("[T1] No fine-tuned model found. Using Detoxify baseline.")
                self.detoxify_model = Detoxify("original")
        except Exception as e:
            print(f"[T1] Model load error: {e}. Using Detoxify.")
            self.detoxify_model = Detoxify("original")

    def _classify_with_finetuned(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        labels = ["safe", "harassment", "threat", "hate_speech", "identity_hate"]
        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
        top_category = max(scores, key=scores.get)
        return {
            "toxicity_score": 1.0 - scores.get("safe", 0.0),
            "top_category": top_category,
            "all_scores": scores,
            "confidence": float(scores[top_category]),
        }

    def _classify_with_detoxify(self, text: str) -> dict:
        results = self.detoxify_model.predict(text)
        toxicity_score = float(results["toxicity"])
        all_scores = {
            "safe": 1.0 - toxicity_score,
            "harassment": float(results["obscene"]),
            "threat": float(results["threat"]),
            "insult": float(results["insult"]),
            "identity_hate": float(results["identity_hate"]),
        }
        top_category = max(all_scores, key=all_scores.get)
        return {
            "toxicity_score": toxicity_score,
            "top_category": top_category if top_category != "safe" else "harassment",
            "all_scores": all_scores,
            "confidence": toxicity_score,
        }

    def _get_severity(self, score: float) -> str:
        if score < 0.40:
            return "none"
        elif score < 0.65:
            return "low"
        elif score < 0.80:
            return "medium"
        return "high"

    def _check_women_risk(self, text: str) -> bool:
        keywords = [
            "rape", "molest", "stalk", "slut", "whore", "bitch",
            "kitchen", "sandwich", "body", "curves", "sexy", "send pics",
            "nudes", "leaked", "expose", "find you", "know where you live",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    async def analyze(self, text: str) -> dict:
        try:
            if self.model is not None:
                result = self._classify_with_finetuned(text)
            else:
                result = self._classify_with_detoxify(text)
            severity = self._get_severity(result["toxicity_score"])
            women_risk = self._check_women_risk(text)
            return {
                "agent": "T1_toxicity",
                "score": result["toxicity_score"],
                "category": result["top_category"],
                "severity": severity,
                "all_scores": result["all_scores"],
                "confidence": result["confidence"],
                "women_risk_flag": women_risk,
                "explanation": f"Toxicity score {result['toxicity_score']:.2f} — {result['top_category']}",
            }
        except Exception as e:
            print(f"[T1] Analysis error: {e}")
            return {
                "agent": "T1_toxicity",
                "score": 0.0,
                "category": "safe",
                "severity": "none",
                "all_scores": {},
                "confidence": 0.0,
                "women_risk_flag": False,
                "explanation": "Analysis failed — defaulting to safe",
            }
