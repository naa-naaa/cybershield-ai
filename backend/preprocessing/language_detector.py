import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 42


class LanguageDetector:
    """
    Detects: tamil_script, hindi_script, telugu_script, tanglish, english, mixed
    """

    TAMIL_PATTERN = re.compile(r"[\u0B80-\u0BFF]")
    HINDI_PATTERN = re.compile(r"[\u0900-\u097F]")
    TELUGU_PATTERN = re.compile(r"[\u0C00-\u0C7F]")
    KANNADA_PATTERN = re.compile(r"[\u0C80-\u0CFF]")
    MALAYALAM_PATTERN = re.compile(r"[\u0D00-\u0D7F]")

    TANGLISH_MARKERS = {
        "machi", "machan", "da", "di", "dei", "enna", "yenna",
        "seri", "po", "va", "vaa", "illa", "illaya", "avan",
        "aval", "nee", "naan", "sollu", "paru", "paaru",
        "saptiya", "sapdiya", "saptu", "vantiya", "vandiya",
        "ennada", "ennadi", "loosu", "poda", "podi",
        "mokka", "vera", "yaar", "yaaru", "dei", "naaye",
    }

    def detect_script(self, text: str):
        if self.TAMIL_PATTERN.search(text):
            return "tamil_script", "tamil"
        if self.HINDI_PATTERN.search(text):
            return "hindi_script", "hindi"
        if self.TELUGU_PATTERN.search(text):
            return "telugu_script", "telugu"
        if self.KANNADA_PATTERN.search(text):
            return "kannada_script", "kannada"
        if self.MALAYALAM_PATTERN.search(text):
            return "malayalam_script", "malayalam"
        return None, None

    def is_tanglish(self, text: str) -> bool:
        words = re.findall(r"\b\w+\b", text.lower())
        return sum(1 for w in words if w in self.TANGLISH_MARKERS) >= 1

    def detect(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {"language": "unknown", "script": None, "needs_translation": False, "needs_tanglish_norm": False}

        script, lang = self.detect_script(text)
        if script:
            return {
                "language": lang,
                "script": script,
                "needs_translation": True,
                "needs_tanglish_norm": False,
            }

        if self.is_tanglish(text):
            return {
                "language": "tanglish",
                "script": "latin",
                "needs_translation": False,
                "needs_tanglish_norm": True,
            }

        try:
            detected = detect(text)
        except LangDetectException:
            detected = "en"

        if detected == "ta":
            return {
                "language": "tanglish",
                "script": "latin",
                "needs_translation": False,
                "needs_tanglish_norm": True,
            }

        return {
            "language": detected,
            "script": "latin",
            "needs_translation": detected not in ["en"],
            "needs_tanglish_norm": False,
        }
