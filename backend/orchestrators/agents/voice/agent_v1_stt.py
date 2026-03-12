import whisper
import tempfile
import os


class AgentV1STT:
    """Speech to Text using OpenAI Whisper"""

    def __init__(self):
        print("[V1] Loading Whisper model...")
        self.model = whisper.load_model("base")
        print("[V1] Whisper loaded")

    async def analyze(self, audio_bytes: bytes) -> dict:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            result = self.model.transcribe(tmp_path, language="en")
            os.unlink(tmp_path)
            return {
                "agent": "V1_stt",
                "transcript": result["text"].strip(),
                "language": result.get("language", "en"),
                "confidence": 0.90,
            }
        except Exception as e:
            return {"agent": "V1_stt", "transcript": "", "language": "en", "confidence": 0.0, "error": str(e)}
