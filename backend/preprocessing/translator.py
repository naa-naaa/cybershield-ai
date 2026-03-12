import torch


class IndianLanguageTranslator:
    """
    Translates Indian language scripts to English.
    Primary: IndicTrans2 (AI4Bharat).
    Fallback: Helsinki-NLP MarianMT.
    """

    SUPPORTED_LANG_CODES = {
        "tamil": "tam_Taml",
        "hindi": "hin_Deva",
        "telugu": "tel_Telu",
        "kannada": "kan_Knda",
        "malayalam": "mal_Mlym",
        "bengali": "ben_Beng",
    }

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model_name = "ai4bharat/indictrans2-indic-en-1B"
            print(f"[Translator] Loading IndicTrans2...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            print("[Translator] IndicTrans2 loaded")
        except Exception as e:
            print(f"[Translator] IndicTrans2 failed: {e}. Trying Helsinki fallback...")
            self._load_fallback()

    def _load_fallback(self):
        try:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = "Helsinki-NLP/opus-mt-dra-en"
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("[Translator] Helsinki fallback loaded")
        except Exception as e:
            print(f"[Translator] Both models failed: {e}. Translation disabled.")
            self.model = None
            self.tokenizer = None

    def translate(self, text: str, source_lang: str = "tamil") -> dict:
        if not text.strip():
            return {"translated_text": text, "confidence": 0.0}
        if self.model is None:
            return {"translated_text": text, "confidence": 0.0, "method": "no_model"}
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=256, num_beams=4)
            translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return {"translated_text": translated, "source_language": source_lang, "confidence": 0.90}
        except Exception as e:
            return {"translated_text": text, "confidence": 0.0, "error": str(e)}
