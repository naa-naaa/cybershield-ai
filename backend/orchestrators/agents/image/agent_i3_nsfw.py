from nudenet import NudeDetector
import tempfile
import os


class AgentI3NSFW:
    """NSFW detection using NudeNet"""

    def __init__(self):
        print("[I3] Loading NudeNet...")
        self.detector = NudeDetector()
        print("[I3] NudeNet ready")

    async def analyze(self, image_bytes: bytes) -> dict:
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(image_bytes)
                tmp_path = f.name
            results = self.detector.detect(tmp_path)
            os.unlink(tmp_path)

            is_explicit = len(results) > 0
            max_score = max((r.get("score", 0) for r in results), default=0.0)

            return {
                "agent": "I3_nsfw",
                "score": float(max_score),
                "is_explicit": is_explicit,
                "detections": results,
                "women_risk_flag": is_explicit,
                "explanation": "Explicit content detected" if is_explicit else "No explicit content",
            }
        except Exception as e:
            return {
                "agent": "I3_nsfw",
                "score": 0.0,
                "is_explicit": False,
                "detections": [],
                "women_risk_flag": False,
                "error": str(e),
            }
