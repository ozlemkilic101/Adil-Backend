from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
import random
import requests

load_dotenv()

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"

# --- Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class PetitionRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    reply: str


SYSTEM_PROMPT_CHAT = """
Sen 'Adil' isimli bir hukuk bilgi asistanısın.

- Türkiye’de geçerli genel hukuk bilgisi ve iş, tüketici, kira, aile hukuku gibi alanlarda KAVRAMSAL açıklamalar yaparsın.
- Kullanıcıya asla 'kesin şu davayı aç' gibi kesin talimat vermezsin, sadece genel bilgi ve olası başvuru yollarını anlatırsın.
- Her cevapta bunun genel bilgilendirme olduğunu, somut olay için bir avukata danışması gerektiğini nazikçe hatırlat.
- Dilin sade, anlaşılır, resmi ve Türkçe olacak.
- Gerektiğinde maddeler halinde, adım adım anlat.
""".strip()

SYSTEM_PROMPT_PETITION = """
Sen Türk mahkemelerine sunulacak RESMİ DİLEKÇE üreten bir asistansın.

KESİNLİKLE UYULACAK KURALLAR:
- SADECE düz metin yaz.
- Markdown, *, -, • gibi işaretler KULLANMA.
- Liste, madde işareti, otomatik numaralandırma YAPMA. (Kullanıcının şablonundaki 1) 2) 3) hariç)
- Metnin sonuna açıklama, uyarı, bilgilendirme EKLEME.
- "Avukata danışınız" vb. ifadeler YAZMA.
- Kullanıcının verdiği şablonu ve sırayı BOZMA.
- Şablon dışına tek cümle ekleme.
""".strip()


# --- HTTP session (keep-alive) ---
_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})

# timeouts (connect, read)
# dilekçe üretimi uzun sürebilir -> read'i büyüt
DEFAULT_TIMEOUT = (10, 140)


def _post_with_retries(url: str, payload: dict, timeout=DEFAULT_TIMEOUT, max_retries: int = 5) -> requests.Response:
    """
    Gemini'ye sağlam istek:
    - 429 / 5xx / timeout / connection reset durumlarında retry
    - exponential backoff + jitter
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            r = _session.post(url, json=payload, timeout=timeout)

            # 200 OK
            if r.status_code == 200:
                return r

            # Rate limit / server issues -> retry
            if r.status_code in (408, 429, 500, 502, 503, 504):
                # Retry-After varsa ona uyalım
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    sleep_s = float(retry_after)
                else:
                    # exponential backoff + jitter
                    base = 1.5 ** attempt
                    sleep_s = min(30.0, 2.0 * base) + random.uniform(0.0, 0.8)

                if attempt < max_retries:
                    time.sleep(sleep_s)
                    continue

            # diğer HTTP hataları: retry yapmadan dön
            return r

        except requests.Timeout as e:
            last_exc = e
            if attempt < max_retries:
                sleep_s = min(35.0, 2.5 * (1.6 ** attempt)) + random.uniform(0.0, 1.2)
                time.sleep(sleep_s)
                continue
            raise

        except requests.RequestException as e:
            last_exc = e
            if attempt < max_retries:
                sleep_s = min(35.0, 2.5 * (1.6 ** attempt)) + random.uniform(0.0, 1.2)
                time.sleep(sleep_s)
                continue
            raise

    # normalde buraya düşmez ama garanti
    raise last_exc or RuntimeError("Gemini request failed")


def call_gemini(
    text: str,
    system_instruction: str | None = None,
    *,
    max_output_tokens: int = 900,
    temperature: float = 0.3,
    timeout=DEFAULT_TIMEOUT
) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing (env var)")

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": text}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"

    try:
        r = _post_with_retries(url, payload, timeout=timeout, max_retries=5)
    except requests.Timeout:
        # Render tarafında kullanıcıya daha açıklayıcı dönelim
        raise HTTPException(
            status_code=504,
            detail="Gemini timeout: yanıt süresi aşıldı. (Uzun metinlerde olabilir; tekrar deneyin.)"
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=504, detail=f"Gemini request failed: {str(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini HTTP {r.status_code}: {r.text}")

    try:
        data = r.json()
        cand = (data.get("candidates") or [{}])[0]
        finish = cand.get("finishReason", "UNKNOWN")
        out = (
            ((cand.get("content") or {}).get("parts") or [{}])[0]
            .get("text", "")
        ).strip()

        if not out:
            raise HTTPException(status_code=502, detail=f"Gemini empty output. finishReason={finish}. Raw={data}")

        # Eğer token yüzünden kesildiyse bunu da bildir (debug için)
        # (İstersen bunu response'a ekleyebiliriz; şimdilik log gibi kalsın)
        # print("Gemini finishReason:", finish, "len:", len(out))

        return out

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=f"Parse error. Raw JSON: {r.text}")


@app.get("/")
def root():
    return {"message": "Adil backend çalışıyor 🚀", "provider": "gemini", "model": GEMINI_MODEL}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    convo_lines = [f"SİSTEM:\n{SYSTEM_PROMPT_CHAT}\n"]
    for m in req.messages:
        role = (m.role or "").strip().lower()
        if role not in ("user", "assistant"):
            role = "user"
        tag = "KULLANICI" if role == "user" else "ASİSTAN"
        convo_lines.append(f"{tag}: {m.content}")

    final_text = "\n\n".join(convo_lines).strip()

    # chat genelde kısa -> 900 token yeter
    reply = call_gemini(final_text, max_output_tokens=900, temperature=0.3)
    return ChatResponse(reply=reply)


@app.post("/api/petitions", response_model=ChatResponse)
def generate_petition(req: PetitionRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    # dilekçe uzun -> token ve read timeout artır
    reply = call_gemini(
        text=prompt,
        system_instruction=SYSTEM_PROMPT_PETITION,
        max_output_tokens=3000,
        temperature=0.2,
        timeout=(10, 180)  # uzun metin için daha güvenli
    )
    return ChatResponse(reply=reply)
