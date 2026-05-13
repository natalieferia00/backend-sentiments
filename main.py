from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from textblob import TextBlob
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from motor.motor_asyncio import AsyncIOMotorClient
import datetime
import nltk  # <--- IMPORTA NLTK

# Descarga los recursos necesarios para TextBlob en el servidor de Render
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# 1. DEFINIR LA APP PRIMERO
app = FastAPI()

# 2. CONFIGURAR MONGODB ATLAS
MONGO_URL = "mongodb+srv://natalieferia1122:1003498135@cluster0.27dplvb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URL)
db = client.sentiment_db
collection = db.history

# 3. CONFIGURAR CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str

# 4. ENDPOINTS
@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(request.text)
        blob = TextBlob(translated)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.05:
            cat, col = "Positivo", "#10B981"
        elif polarity < -0.05:
            cat, col = "Negativo", "#EF4444"
        else:
            cat, col = "Neutro", "#6B7280"

        result = {
            "text": request.text,
            "sentiment": cat,
            "score": round(polarity, 2),
            "color": col,
            "date": datetime.datetime.now().strftime("%H:%M:%S")
        }
        
        await collection.insert_one(result.copy())
        return result
    except Exception as e:
        print(f"Error crítico en analyze: {e}")
        # Retornamos explícitamente un estado HTTP 500 para que no se disfrace de error de red
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    try:
        cursor = collection.find().sort("_id", -1).limit(5)
        history = await cursor.to_list(length=5)
        for item in history:
            item["_id"] = str(item["_id"])
        return history
    except Exception as e:
        print(f"Error en history: {e}")
        return []