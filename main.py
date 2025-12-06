from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

# Groq istemcisi
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# Android'den (veya Postman'den) alacağımız veri modeli
class ChatMessage(BaseModel):
    role: str  # "user" veya "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class ChatResponse(BaseModel):
    reply: str


SYSTEM_PROMPT = """
Sen 'Adil' isimli bir hukuk bilgi asistanısın.

- Türkiye’de geçerli genel hukuk bilgisi ve iş, tüketici, kira, aile hukuku gibi alanlarda KAVRAMSAL açıklamalar yaparsın.
- Kullanıcıya asla 'kesin şu davayı aç' gibi kesin talimat vermezsin, sadece genel bilgi ve olası başvuru yollarını anlatırsın.
- Her cevapta bunun genel bilgilendirme olduğunu, somut olay için bir avukata danışması gerektiğini nazikçe hatırlat.
- Dilin sade, anlaşılır, resmi ve Türkçe olacak.
- Gerektiğinde maddeler halinde, adım adım anlat.
"""


@app.get("/")
def root():
    return {"message": "Adil backend çalışıyor 🚀"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Genel hukuk chatbot endpoint'i.
    Android buraya messages listesini gönderir.
    """

    # Groq'a gidecek mesaj listesi
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    # Groq LLM çağrısı
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # hızlı ve güçlü bir model
        messages=messages,
        max_tokens=512,
        temperature=0.3,  # çok yaratıcı olmasın, daha tutarlı olsun
    )

    reply_text = completion.choices[0].message.content
    return ChatResponse(reply=reply_text)
