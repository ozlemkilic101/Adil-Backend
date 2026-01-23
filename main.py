from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # ister flash-lite / pro yap
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"

# Android'den (veya Postman'den) alacağımız veri modeli
class ChatMessage(BaseModel):
    role: str  # "user" veya "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class PetitionRequest(BaseModel):
    prompt: str  # Android'den buildPrompt() ile gelen TAM METİN

class ChatResponse(BaseModel):
    reply: str


# ✅ Genel hukuk chatbot prompt'u (disclaimer'lı - sohbet için OK)
SYSTEM_PROMPT_CHAT = """
Sen 'Adil' isimli bir hukuk bilgi asistanısın.

- Türkiye’de geçerli genel hukuk bilgisi ve iş, tüketici, kira, aile hukuku gibi alanlarda KAVRAMSAL açıklamalar yaparsın.
- Kullanıcıya asla 'kesin şu davayı aç' gibi kesin talimat vermezsin, sadece genel bilgi ve olası başvuru yollarını anlatırsın.
- Her cevapta bunun genel bilgilendirme olduğunu, somut olay için bir avukata danışması gerektiğini nazikçe hatırlat.
- Dilin sade, anlaşılır, resmi ve Türkçe olacak.
- Gerektiğinde maddeler halinde, adım adım anlat.
""".strip()

# ✅ Dilekçe prompt'u (disclaimer YOK)
SYSTEM_PROMPT_PETITION = """
Sen Türk mahkemelerine sunulacak RESMİ DİLEKÇE üreten bir asistansın.

KESİNLİKLE UYULACAK KURALLAR:
- SADECE düz metin yaz.
- Markdown, *, -, • gibi işaretler KULLANMA.
- Liste, madde işareti, otomatik numaralandırma YAPMA.
- Metnin sonuna açıklama, uyarı, bilgilendirme EKLEME.
- "Avukata danışınız" vb. ifadeler YAZMA.
- Kullanıcının verdiği şablonu ve sırayı BOZMA.
- Şablon dışına tek cümle ekleme.
""".strip()


def call_gemini(text: str, system_instruction: str | None = None) -> str:
    """
    Gemini generateContent çağrısı.
    text: kullanıcı + bağlam mesajları (tek text olarak)
    system_instruction: istersek ayrı 'system' talimatı (format disiplini için iyi)
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing (env var)")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": text}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 800
        }
    }

    # Gemini systemInstruction destekler
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    try:
        r = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=(10, 90)
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=504, detail=f"Gemini request failed: {str(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini HTTP {r.status_code}: {r.text}")

    try:
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise HTTPException(status_code=500, detail=f"Parse error. Raw JSON: {r.text}")


@app.get("/")
def root():
    return {"message": "Adil backend çalışıyor 🚀", "provider": "gemini", "model": GEMINI_MODEL}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Genel hukuk chatbot endpoint'i.
    Android buraya messages listesini gönderir.
    """
    # Groq'ta system+messages vardı -> burada tek text'e çeviriyoruz
    # Basit ve sağlam yöntem: tüm konuşmayı tek metin haline getir
    convo_lines = [f"SİSTEM:\n{SYSTEM_PROMPT_CHAT}\n"]
    for m in req.messages:
        role = (m.role or "").strip().lower()
        if role not in ("user", "assistant"):
            role = "user"
        tag = "KULLANICI" if role == "user" else "ASİSTAN"
        convo_lines.append(f"{tag}: {m.content}")

    final_text = "\n\n".join(convo_lines).strip()

    # İstersen burada systemInstruction da geçebilirsin ama text içinde zaten var
    reply = call_gemini(final_text)
    return ChatResponse(reply=reply)


@app.post("/api/petitions", response_model=ChatResponse)
def generate_petition(req: PetitionRequest):
    """
    Android DilekceViewModel -> buildPrompt() çıktısını
    DOĞRUDAN buraya gönderir.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    reply = call_gemini(
        text=prompt,
        system_instruction=SYSTEM_PROMPT_PETITION
    )
    return ChatResponse(reply=reply)
