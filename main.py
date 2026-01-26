from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
import time
import random
from enum import Enum

load_dotenv()

app = FastAPI()

# ---------------------------
# ENV
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_CHAT = os.getenv("GROQ_MODEL_CHAT", "llama-3.1-8b-instant")
GROQ_MODEL_PETITION = os.getenv("GROQ_MODEL_PETITION", "llama-3.1-70b-versatile")

# Dilekçe üretimi için güvenli limitler (Render timeout yememek için)
PETITION_DRAFT_MAX_TOKENS = int(os.getenv("PETITION_DRAFT_MAX_TOKENS", "900"))
PETITION_IMPROVE_MAX_TOKENS = int(os.getenv("PETITION_IMPROVE_MAX_TOKENS", "650"))
PETITION_MIN_CHARS_FOR_SKIP_IMPROVE = int(os.getenv("PETITION_MIN_CHARS_FOR_SKIP_IMPROVE", "1400"))

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Render Env Vars'a eklemelisin.")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Schemas
# ---------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class DilekceType(str, Enum):
    ALACAK_DAVASI = "ALACAK_DAVASI"
    CEVAP_DILEKCESI = "CEVAP_DILEKCESI"
    ITIRAZ_DILEKCESI = "ITIRAZ_DILEKCESI"
    ICRA_ITIRAZ = "ICRA_ITIRAZ"
    SIKAYET = "SIKAYET"
    TAZMINAT = "TAZMINAT"
    DIGER = "DIGER"

class PetitionRequest(BaseModel):
    dilekce_tipi: DilekceType
    prompt: str
    diger_aciklama: str | None = None

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
    base = min(6.0, (1.6 ** attempt))
    time.sleep(0.8 + base + random.uniform(0.0, 0.6))

def call_groq(messages: list[dict], *, model: str, max_tokens: int, temperature: float, retries: int = 1) -> str:
    """
    Groq çağrısı + retry.
    Render timeout riskini azaltmak için retries varsayılanı 1'e çekildi.
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
    s = raw.strip()
    s = s.replace("**", "").replace("__", "").replace("`", "")

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

    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")

    return s

def build_petition_user_prompt(req: PetitionRequest) -> str:
    tip_line = f"SEÇİLEN DİLEKÇE TÜRÜ: {req.dilekce_tipi.value}"

    extra = ""
    if req.dilekce_tipi == DilekceType.DIGER:
        extra = f"\nDİĞER AÇIKLAMA: {req.diger_aciklama.strip()}"

    return f"{tip_line}{extra}\n\n{req.prompt.strip()}"

def build_improve_prompt(draft_text: str) -> str:
    # İkinci çağrı: sadece mevcut dilekçeyi aynı formatla biraz daha detaylandır
    return (
        "Aşağıdaki dilekçeyi şablon sırasını ve resmi dili koruyarak biraz daha detaylandır. "
        "Yeni başlık ekleme, markdown kullanma, en sona uyarı ekleme.\n\n"
        f"{draft_text}"
    )

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
        "petition_draft_max_tokens": PETITION_DRAFT_MAX_TOKENS,
        "petition_improve_max_tokens": PETITION_IMPROVE_MAX_TOKENS,
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    for m in req.messages:
        role = (m.role or "").strip().lower()
        if role not in ("user", "assistant"):
            role = "user"
        msgs.append({"role": role, "content": (m.content or "").strip()})

    reply = call_groq(
        msgs,
        model=GROQ_MODEL_CHAT,
        max_tokens=900,
        temperature=0.3,
        retries=2
    )
    return ChatResponse(reply=reply)

@app.post("/api/petitions", response_model=ChatResponse)
def generate_petition(req: PetitionRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    if req.dilekce_tipi == DilekceType.DIGER:
        if not req.diger_aciklama or not req.diger_aciklama.strip():
            raise HTTPException(status_code=400, detail="diger_aciklama is required when dilekce_tipi is DIGER")

    user_prompt = build_petition_user_prompt(req)

    # 1) DRAFT: hızlı model ile (timeout yememek için)
    draft_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_PETITION},
        {"role": "user", "content": user_prompt}
    ]

    draft = call_groq(
        draft_msgs,
        model=GROQ_MODEL_CHAT,                  # ✅ hızlı model
        max_tokens=PETITION_DRAFT_MAX_TOKENS,
        temperature=0.2,
        retries=1
    )
    draft = normalize_petition_text(draft)

    # 2) IMPROVE: sadece gerekiyorsa (kısa kaldıysa) 70B ile küçük revizyon
    if len(draft) < PETITION_MIN_CHARS_FOR_SKIP_IMPROVE:
        improve_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_PETITION},
            {"role": "user", "content": build_improve_prompt(draft)}
        ]
        improved = call_groq(
            improve_msgs,
            model=GROQ_MODEL_PETITION,          # ✅ kalite için
            max_tokens=PETITION_IMPROVE_MAX_TOKENS,
            temperature=0.2,
            retries=1
        )
        improved = normalize_petition_text(improved)
        if len(improved) > len(draft):
            draft = improved

    return ChatResponse(reply=draft)
