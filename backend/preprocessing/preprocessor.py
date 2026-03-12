import re
from preprocessing.language_detector import LanguageDetector
from preprocessing.translator import IndianLanguageTranslator
from preprocessing.tanglish_normalizer import TanglishNormalizer
from preprocessing.emoji_interpreter import expand_emojis, has_emoji


class TextPreprocessor:
    """
    Master preprocessor — runs before all agents.
    Pipeline: Emoji expansion → Language detection → Translation/Normalization → Cleanup
    Agents always receive clean English text.
    """

    def __init__(self):
        print("[Preprocessor] Initializing...")
        self.detector = LanguageDetector()
        self.translator = IndianLanguageTranslator()
        self.tanglish = TanglishNormalizer()
        print("[Preprocessor] Ready")

    def process(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "processed_text": "",
                "original_text": text,
                "language_detected": "unknown",
                "transformations": [],
                "preprocessing_boost": 0.0,
            }

        original = text
        transformations = []
        preprocessing_boost = 0.0

        # Step 1 — Expand emojis
        if has_emoji(text):
            text = expand_emojis(text)
            transformations.append("emoji_expanded")

        # Step 2 — Detect language
        lang_info = self.detector.detect(text)
        language = lang_info["language"]

        # Step 3 — Script-based translation (Tamil/Hindi/Telugu/etc.)
        if lang_info.get("needs_translation"):
            result = self.translator.translate(text, source_lang=language)
            text = result["translated_text"]
            transformations.append(f"translated_from_{language}")

        # Step 4 — Tanglish: romanized → native script → translate
        elif lang_info.get("needs_tanglish_norm"):
            script_text = self.tanglish.romanized_to_script(text, lang_code="ta")
            result = self.translator.translate(script_text, source_lang="tamil")
            text = result["translated_text"]
            transformations.append("tanglish_via_xlit_translation")
            preprocessing_boost += 0.02  # small boost for slang-heavy content

        # Step 5 — Cleanup
        text = re.sub(r"\s+", " ", text).strip()

        return {
            "processed_text": text,
            "original_text": original,
            "language_detected": language,
            "transformations": transformations,
            "preprocessing_boost": min(preprocessing_boost, 0.20),
        }
