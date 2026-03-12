class TanglishNormalizer:
    """
    Tanglish → Tamil Script (IndicXlit) → English (IndicTrans2).
    No LLM. No Gemma. Pure transliteration pipeline.
    """

    def __init__(self):
        self.engines = {}
        print("[Tanglish] Loading IndicXlit engines...")
        try:
            from ai4bharat.transliteration import XlitEngine
            for lang_code in ["ta", "hi", "te", "kn", "ml"]:
                try:
                    self.engines[lang_code] = XlitEngine(lang_code)
                    print(f"[Tanglish] IndicXlit loaded for {lang_code}")
                except Exception as e:
                    print(f"[Tanglish] Could not load {lang_code}: {e}")
        except ImportError:
            print("[Tanglish] ai4bharat-transliteration not installed. Install with:")
            print("  pip install ai4bharat-transliteration")

    def romanized_to_script(self, text: str, lang_code: str = "ta") -> str:
        """
        Convert romanized Indian language words to native script.
        'saptiya' → 'சாப்பிட்டியா'
        Then IndicTrans2 can translate the native script properly.
        """
        if lang_code not in self.engines:
            return text  # Return as-is if engine not available
        engine = self.engines[lang_code]
        words = text.split()
        converted = []
        for word in words:
            try:
                result = engine.translit_word(word, topk=1)
                if result and result[0]:
                    converted.append(result[0])
                else:
                    converted.append(word)
            except Exception:
                converted.append(word)
        return " ".join(converted)
