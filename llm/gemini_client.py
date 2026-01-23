import os
import httpx
from typing import Optional

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # istersen değiştir

class GeminiError(RuntimeError):
    pass

async def gemini_generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    timeout_sec: float = 60.0,
    system_instruction: Optional[str] = None,
) -> str:
    if not GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is missing")

    url = f"{GEMINI_BASE_URL}/v1/models/{model}:generateContent"
    params = {"key": GEMINI_API_KEY}

    # Gemini request format: contents -> parts -> text
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    # İstersen sistem talimatı ekleyebilirsin (format disiplinine yardımcı olur)
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        r = await client.post(url, params=params, json=payload)
        if r.status_code != 200:
            raise GeminiError(f"Gemini API error {r.status_code}: {r.text}")

        data = r.json()

    # Çıktı genelde: candidates[0].content.parts[0].text
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise GeminiError(f"Unexpected Gemini response: {data}")
