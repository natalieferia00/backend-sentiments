from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from textblob import TextBlob
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from motor.motor_asyncio import AsyncIOMotorClient
import datetime
import nltk

# Descarga los recursos necesarios para TextBlob en el servidor de Render
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# 1. DEFINIR LA APP
app = FastAPI()

# 2. CONFIGURAR MONGODB ATLAS
MONGO_URL = "mongodb+srv://natalieferia1122:1003498135@cluster0.27dplvb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URL)
db = client.sentiment_db
collection = db.history

# 3. CONFIGURAR CORS PLANO (Sin credenciales para que acepte el asterisco)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str

# 4. ENDPOINT ÚNICO PARA ANALIZAR PALABRAS
@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    try:
        # Traducción e IA
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
        
        # Guardar registro en Atlas
        await collection.insert_one(result.copy())
        
        return result
    except Exception as e:
        print(f"Error en analyze: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

# 5. NUEVO ENDPOINT: CONSULTAR EL HISTORIAL DE MONGODB
@app.get("/history")
async def get_history():
    try:
        # Busca los últimos 10 registros guardados, ordenados del más reciente al más antiguo
        cursor = collection.find().sort("_id", -1).limit(10)
        history_list = []
        
        async for document in cursor:
            # Convertimos el ObjectId de MongoDB a string para evitar errores de serialización JSON
            document["_id"] = str(document["_id"])
            history_list.append(document)
            
        return history_list
    except Exception as e:
        print(f"Error en history: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))