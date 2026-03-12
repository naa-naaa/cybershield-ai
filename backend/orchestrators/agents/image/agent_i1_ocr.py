import pytesseract
from PIL import Image
import io


class AgentI1OCR:
    """Extract text from images using pytesseract"""

    def __init__(self):
        print("[I1] OCR agent ready")

    async def analyze(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            extracted_text = pytesseract.image_to_string(image).strip()
            return {
                "agent": "I1_ocr",
                "extracted_text": extracted_text,
                "has_text": len(extracted_text) > 3,
                "char_count": len(extracted_text),
            }
        except Exception as e:
            return {"agent": "I1_ocr", "extracted_text": "", "has_text": False, "char_count": 0, "error": str(e)}
