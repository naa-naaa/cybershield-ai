import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np
import tempfile
import os


class AgentV3VoiceEmotion:
    """
    Voice Emotion Recognition using wav2vec2.
    Fine-tuned on RAVDESS + CREMA-D (or uses pretrained baseline).
    """

    def __init__(self):
        self.model_path = os.path.join(os.getenv("MODEL_PATH", "./models"), "voice_emotion")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.feature_extractor = None
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path) and os.listdir(self.model_path):
                print(f"[V3] Loading fine-tuned voice emotion from {self.model_path}")
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path).to(self.device)
                self.model.eval()
            else:
                print("[V3] Loading baseline wav2vec2 emotion model...")
                model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(self.device)
                self.model.eval()
                print("[V3] Baseline voice emotion model loaded")
        except Exception as e:
            print(f"[V3] Model load error: {e}. Voice emotion scoring disabled.")

    async def analyze(self, audio_bytes: bytes) -> dict:
        if self.model is None:
            return {
                "agent": "V3_voice_emotion",
                "score": 0.0,
                "emotion": "unknown",
                "anger_probability": 0.0,
                "fear_probability": 0.0,
                "distress_score": 0.0,
                "explanation": "Model unavailable",
            }
        try:
            import librosa
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            y, sr = librosa.load(tmp_path, sr=16000)
            os.unlink(tmp_path)

            inputs = self.feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            labels = self.model.config.id2label
            emotion_scores = {labels[i]: float(probs[i]) for i in range(len(probs))}
            dominant = max(emotion_scores, key=emotion_scores.get)

            anger_prob = emotion_scores.get("angry", emotion_scores.get("anger", 0.0))
            fear_prob = emotion_scores.get("fearful", emotion_scores.get("fear", 0.0))
            distress_score = (anger_prob + fear_prob) / 2

            return {
                "agent": "V3_voice_emotion",
                "score": distress_score,
                "emotion": dominant,
                "anger_probability": anger_prob,
                "fear_probability": fear_prob,
                "distress_score": distress_score,
                "all_scores": emotion_scores,
                "explanation": f"Dominant emotion: {dominant}, distress: {distress_score:.2f}",
            }
        except Exception as e:
            return {
                "agent": "V3_voice_emotion",
                "score": 0.0,
                "emotion": "unknown",
                "anger_probability": 0.0,
                "fear_probability": 0.0,
                "distress_score": 0.0,
                "error": str(e),
            }
