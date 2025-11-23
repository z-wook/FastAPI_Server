import os
import base64
import httpx
from typing import Union
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, wait_fixed, stop_after_delay

load_dotenv()
app = FastAPI()

HF_API_KEY = os.getenv("HF_TOKEN")
API_URL = (
    "https://router.huggingface.co/fal-ai/fal-ai/flux-kontext/dev?_subdomain=queue"
)

# --------------------------
# 요청 Body 모델
# --------------------------
class ImageRequest(BaseModel):
    image_base64: str
    prompt: str


# --------------------------
# HF API 초기 요청
# --------------------------
async def send_hf_request(client: httpx.AsyncClient, image_base64: str, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": image_base64, "parameters": {"prompt": prompt}}
    
    response = await client.post(url=API_URL, headers=headers, json=payload)
    print("\n🔥🔥🔥 HF 응답 상태코드:", response.status_code)
    print("🔥🔥🔥 HF 응답 바디:", response.text, "\n")

    if response.status_code != httpx.codes.OK:
        raise Exception(f"HF API 호출 실패: {response.text}")
    
    data = response.json()
    status_url = data.get("status_url")
    
    if not status_url:
        raise Exception(f"status_url 없음: {data}")
    return status_url


# --------------------------
# Polling
# --------------------------
async def poll_status(client: httpx.AsyncClient, status_url: str) -> dict:
    async for attempt in AsyncRetrying(
        wait=wait_fixed(1),
        stop=stop_after_delay(60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            response = await client.get(status_url)
            data = response.json()
            status = data.get("status")
            if status == "COMPLETED":
                return data
            elif status == "FAILED":
                raise Exception(f"HF 이미지 생성 실패: {data}")
            else:
                raise Exception("진행 중...")


# --------------------------
# 이미지 가져오기
# --------------------------
async def fetch_image(client: httpx.AsyncClient, url: str) -> bytes:
    res = await client.get(url)
    res.raise_for_status()
    return res.content


# --------------------------
# FastAPI 엔드포인트
# --------------------------
@app.post("/generate")
async def generate_image(data: ImageRequest):
    async with httpx.AsyncClient(timeout=60) as client:
        # 1) HF API 초기 요청
        try:
            status_url = await send_hf_request(client, data.image_base64, data.prompt)
        except Exception as e:
            return {"error": True, "message": "HF API 호출 실패", "details": str(e)}
        
        # 2) Polling으로 완료 대기
        try:
            status_json = await poll_status(client, status_url)
        except Exception as e: 
            return {"error": True, "message": "polling 실패", "details": str(e)}
        
        # 3) 최종 이미지 가져오기
        image_url = status_json.get("result", {}).get("image", {}).get("url")
        if not image_url:
            return {"error": True, "message": "이미지 URL 없음"}
        
        try:
            image_bytes = await fetch_image(client, image_url)
        except Exception as e:
            return {"error": True, "message": "이미지 다운로드 실패", "details": str(e)}
        
    # 4) Base64 변환 후 반환
    result_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return {"result_base64": result_base64}
