import torch
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported direction: "en-indic" or "indic-en"
_models = {}

def _load_model(direction: str):
    if direction not in _models:
        model_id = (
            "ai4bharat/indictrans2-en-indic-dist-200M"
            if direction == "en-indic"
            else "ai4bharat/indictrans2-indic-en-dist-200M"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(DEVICE)
        _models[direction] = (tokenizer, model)
    return _models[direction]


def translate(texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    """
    Translate a list of sentences.

    Language codes follow IndicTrans2 format, e.g.:
      English  -> eng_Latn
      Hindi    -> hin_Deva
      Tamil    -> tam_Taml
      Telugu   -> tel_Telu

    Example:
        translate(["Hello world"], "eng_Latn", "hin_Deva")
    """
    direction = "en-indic" if src_lang == "eng_Latn" else "indic-en"
    tokenizer, model = _load_model(direction)
    ip = IndicProcessor(inference=True)

    batch = ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(**inputs, num_beams=5, max_length=256)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(decoded, lang=tgt_lang)
