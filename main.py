from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
import time
import random

load_dotenv()

app = FastAPI()

# ---------------------------
# ENV
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_CHAT = os.getenv("GROQ_MODEL_CHAT", "llama-3.1-8b-instant")
GROQ_MODEL_PETITION = os.getenv("GROQ_MODEL_PETITION", "llama-3.1-70b-versatile")  # dilekçe için daha iyi

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Render Env Vars'a eklemelisin.")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Schemas
# ---------------------------
class ChatMessage(BaseModel):
    role: str  # "user" veya "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class PetitionRequest(BaseModel):
    prompt: str  # Android buildPrompt() çıktısı

class ChatResponse(BaseModel):
    reply: str

# ---------------------------
# Prompts
# ---------------------------
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
- Şablonun başlık sırasını ve boş satırlarını BOZMA.
- Şablon dışına tek cümle ekleme.
- Metnin sonuna açıklama/uyarı/bilgilendirme ekleme.
""".strip()

# ---------------------------
# Helpers
# ---------------------------
def _backoff_sleep(attempt: int):
    # exponential backoff + jitter (timeout / geçici hata için)
    base = min(20.0, (1.6 ** attempt))
    time.sleep(1.5 + base + random.uniform(0.0, 0.8))

def call_groq(messages: list[dict], *, model: str, max_tokens: int, temperature: float, retries: int = 3) -> str:
    """
    Groq çağrısı + retry.
    - Groq SDK ağ hatalarında exception fırlatabilir; retry ile sağlamlaştırıyoruz.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Empty reply from Groq")
            return text
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            _backoff_sleep(attempt)
    raise HTTPException(status_code=504, detail=f"Groq request failed: {str(last_err)}")

def normalize_petition_text(raw: str) -> str:
    """
    Dilekçede istemediğin markdown/disclaimer vb. çıkarsa temizler.
    (Groq genelde uyuyor ama garanti olsun)
    """
    s = raw.strip()
    # basit markdown temizleme
    s = s.replace("**", "").replace("__", "").replace("`", "")

    # Sonda ekstra bilgilendirme satırlarını kırp
    cut_keywords = [
        "genel bilgilendirme",
        "avukata danış",
        "bilgilendirme amaçlıdır",
        "somut olay",
        "danışmanız önerilir"
    ]
    lines = []
    for line in s.splitlines():
        t = line.strip().lower()
        if any(k in t for k in cut_keywords):
            break
        lines.append(line)
    s = "\n".join(lines).strip()

    # çok fazla boşluk
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")

    return s

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {
        "message": "Adil backend çalışıyor 🚀",
        "provider": "groq",
        "chat_model": GROQ_MODEL_CHAT,
        "petition_model": GROQ_MODEL_PETITION,
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Genel hukuk chatbot endpoint'i.
    Android buraya messages listesini gönderir.
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    for m in req.messages:
        role = (m.role or "").strip().lower()
        if role not in ("user", "assistant"):
            role = "user"
        msgs.append({"role": role, "content": (m.content or "").strip()})

    reply = call_groq(
        msgs,
        model=GROQ_MODEL_CHAT,
        max_tokens=900,        # chat için yeterli
        temperature=0.3,
        retries=3
    )
    return ChatResponse(reply=reply)

@app.post("/api/petitions", response_model=ChatResponse)
def generate_petition(req: PetitionRequest):
    """
    Android DilekceViewModel -> buildPrompt() çıktısını DOĞRUDAN buraya gönderir.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_PETITION},
        {"role": "user", "content": prompt}
    ]

    # dilekçe uzun: token'ı yükselt
    reply = call_groq(
        msgs,
        model=GROQ_MODEL_PETITION,
        max_tokens=2500,       # A4 çıktıya yakın uzunluk
        temperature=0.2,
        retries=3
    )

    reply = normalize_petition_text(reply)
    return ChatResponse(reply=reply)
