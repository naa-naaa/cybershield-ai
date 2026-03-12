import librosa
import numpy as np
import tempfile
import os


class AgentV2Acoustic:
    """Acoustic feature analysis using librosa"""

    def __init__(self):
        print("[V2] Acoustic analyzer ready")

    async def analyze(self, audio_bytes: bytes) -> dict:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            y, sr = librosa.load(tmp_path, sr=None)
            os.unlink(tmp_path)

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

            rms = librosa.feature.rms(y=y)[0]
            avg_volume = float(np.mean(rms))
            volume_spike = float(np.max(rms)) / (avg_volume + 1e-8)

            zcr = librosa.feature.zero_crossing_rate(y)[0]
            speech_rate = float(np.mean(zcr))

            aggression_score = 0.0
            if avg_pitch > 250:
                aggression_score += 0.30
            if volume_spike > 3.0:
                aggression_score += 0.35
            if speech_rate > 0.15:
                aggression_score += 0.20
            if avg_volume > 0.05:
                aggression_score += 0.15

            return {
                "agent": "V2_acoustic",
                "score": float(np.clip(aggression_score, 0.0, 1.0)),
                "avg_pitch": avg_pitch,
                "avg_volume": avg_volume,
                "volume_spike": volume_spike,
                "speech_rate": speech_rate,
                "explanation": f"Pitch:{avg_pitch:.0f}Hz, Volume spike:{volume_spike:.1f}x",
            }
        except Exception as e:
            return {"agent": "V2_acoustic", "score": 0.0, "error": str(e)}
