import os
import base64
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://router.huggingface.co/fal-ai/fal-ai/flux-kontext/dev?_subdomain=queue"

class ImageRequest(BaseModel):
    image_base64: str  # iOS에서 Base64로 이미지를 보내도록 함
    prompt: str

@app.post("/generate")
async def generate_image(data: ImageRequest):

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    # Base64 → raw bytes
    img_bytes = base64.b64decode(data.image_base64)

    # HuggingFace에 보낼 JSON payload
    payload = {
        "inputs": base64.b64encode(img_bytes).decode("utf-8"),  # Base64 인코딩
        "parameters": {
            "prompt": data.prompt
        }
    }

    # HF API 비동기 요청
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return {
            "error": True,
            "message": "HuggingFace API failed",
            "details": response.text
        }

    # 응답 이미지 raw
    image_bytes = response.content

    # raw → Base64 (iOS가 사용하기 쉽도록)
    result_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {"result_base64": result_base64}
