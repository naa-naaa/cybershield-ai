import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import numpy as np


class AgentI2Visual:
    """
    Visual Harm Detection using CLIP.
    Compares image embeddings against harm category prompts.
    """

    HARM_PROMPTS = [
        "a photo showing violence or physical harm",
        "a photo showing threatening gestures or weapons",
        "a photo showing bullying or harassment",
        "a photo showing hate symbols or offensive content",
        "a photo showing humiliation or mockery",
    ]

    SAFE_PROMPTS = [
        "a normal everyday photo",
        "a friendly social media photo",
        "a nature or landscape photo",
    ]

    def __init__(self):
        print("[I2] Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        print("[I2] CLIP loaded")

    async def analyze(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            all_prompts = self.HARM_PROMPTS + self.SAFE_PROMPTS

            inputs = self.processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()

            harm_score = float(np.mean(probs[:len(self.HARM_PROMPTS)]))
            safe_score = float(np.mean(probs[len(self.HARM_PROMPTS):]))

            top_idx = int(np.argmax(probs[:len(self.HARM_PROMPTS)]))
            top_category = self.HARM_PROMPTS[top_idx]

            return {
                "agent": "I2_visual",
                "score": harm_score,
                "visual_harm_score": harm_score,
                "safe_score": safe_score,
                "top_category": top_category,
                "explanation": f"Visual harm probability: {harm_score:.2f}",
            }
        except Exception as e:
            return {"agent": "I2_visual", "score": 0.0, "visual_harm_score": 0.0, "error": str(e)}
