import os
import base64
import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, wait_fixed, stop_after_delay, retry_if_exception_type

load_dotenv()
app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://router.huggingface.co/fal-ai/fal-ai/flux-kontext/dev?_subdomain=queue"

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str

# --------------------------
# Tenacity 기반 polling 함수
# --------------------------
@retry(
    wait=wait_fixed(1),  # 1초 간격
    stop=stop_after_delay(60),  # 최대 60초 대기
    retry=retry_if_exception_type(Exception)  # 예외 발생 시 재시도
)
async def poll_status(client: httpx.AsyncClient, status_url: str) -> dict:
    res = await client.get(status_url)
    data = res.json()
    status = data.get("status")
    if status == "COMPLETED":
        return data
    elif status == "FAILED":
        raise Exception(f"HF 이미지 생성 실패: {data}")
    else:
        # 아직 진행 중이면 예외를 발생시켜 tenacity가 재시도
        raise Exception("이미지 생성 진행 중...")

# --------------------------
# FastAPI 엔드포인트
# --------------------------
@app.post("/generate")
async def generate_image(data: ImageRequest):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    # 1️⃣ Base64 → raw bytes
    try:
        img_bytes = base64.b64decode(data.image_base64)
    except Exception as e:
        return {"error": True, "message": "이미지 Base64 디코딩 실패", "details": str(e)}

    # 2️⃣ HF 요청 payload
    payload = {
        "inputs": base64.b64encode(img_bytes).decode("utf-8"),
        "parameters": {"prompt": data.prompt}
    }

    async with httpx.AsyncClient(timeout=60) as client:
        # 3️⃣ 초기 POST 요청
        init_res = await client.post(API_URL, headers=headers, json=payload)
        if init_res.status_code != 200:
            return {
                "error": True,
                "message": "HuggingFace API 초기 요청 실패",
                "details": init_res.text
            }

        init_data = init_res.json()
        status_url = init_data.get("status_url")
        if not status_url:
            return {
                "error": True,
                "message": "status_url 없음, Queue 모델 확인 필요",
                "details": init_data
            }

        # 4️⃣ Tenacity 기반 polling
        try:
            status_json = await poll_status(client, status_url)
        except Exception as e:
            return {"error": True, "message": "이미지 생성 polling 실패", "details": str(e)}

        # 5️⃣ 완료 → response에서 image URL 가져오기
        result = status_json.get("result", {})
        image_info = result.get("image")
        if not image_info or "url" not in image_info:
            return {"error": True, "message": "이미지 URL 없음", "details": result}

        image_url = image_info["url"]

        # 6️⃣ 이미지 다운로드
        img_res = await client.get(image_url)
        if img_res.status_code != 200:
            return {"error": True, "message": "이미지 다운로드 실패"}

        final_img_bytes = img_res.content

    # 7️⃣ Base64로 인코딩 후 반환
    result_base64 = base64.b64encode(final_img_bytes).decode("utf-8")
    return {"result_base64": result_base64}
