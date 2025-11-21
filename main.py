import os
import base64
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import AsyncRetrying, retry_if_exception_type, wait_fixed, stop_after_delay

load_dotenv()
app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://router.huggingface.co/fal-ai/fal-ai/flux-kontext/dev?_subdomain=queue"

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str

# --------------------------
# ASYNC-TENACITY POLLING
# --------------------------
async def poll_status(client: httpx.AsyncClient, status_url: str) -> dict:
    async for attempt in AsyncRetrying(
        wait=wait_fixed(1),
        stop=stop_after_delay(60),
        retry=retry_if_exception_type(Exception),
        reraise=True
    ):
        with attempt:
            res = await client.get(status_url)
            data = res.json()
            status = data.get("status")

            if status == "COMPLETED":
                return data
            elif status == "FAILED":
                raise Exception(f"HF 이미지 생성 실패: {data}")
            else:
                raise Exception("진행 중...")

# --------------------------
# FastAPI 엔드포인트
# --------------------------
@app.post("/generate")
async def generate_image(data: ImageRequest):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    # Base64 → bytes
    img_bytes = base64.b64decode(data.image_base64)

    payload = {
        "inputs": base64.b64encode(img_bytes).decode(),
        "parameters": {"prompt": data.prompt}
    }

    async with httpx.AsyncClient(timeout=60) as client:
        init_res = await client.post(API_URL, headers=headers, json=payload)

        init_data = init_res.json()
        status_url = init_data.get("status_url")
        if not status_url:
            return {"error": True, "message": "status_url 없음", "details": init_data}

        # Polling (async-safe)
        try:
            status_json = await poll_status(client, status_url)
        except Exception as e:
            return {"error": True, "message": "polling 실패", "details": str(e)}

        # 이미지 URL 확인
        image_url = status_json.get("result", {}).get("image", {}).get("url")
        if not image_url:
            return {"error": True, "message": "이미지 URL 없음"}

        img_res = await client.get(image_url)
        final_bytes = img_res.content

    # 마지막 Base64 변환
    result_base64 = base64.b64encode(final_bytes).decode()
    return {"result_base64": result_base64}
