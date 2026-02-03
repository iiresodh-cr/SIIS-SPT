import os
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import firestore, storage
import vertexai
from vertexai.generative_models import GenerativeModel
import firebase_admin
from firebase_admin import auth

# Configuración de entorno
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
APP_ID = os.getenv("APP_ID", "siis-spt-cr")

app = FastAPI()

# CORS para permitir peticiones desde el frontend local o Cloud Run
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Inicialización de servicios
firebase_admin.initialize_app()
db = firestore.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

async def verify_token(authorization: str = Security(None)):
    if not authorization: raise HTTPException(status_code=401)
    try:
        token = authorization.split("Bearer ")[1]
        return auth.verify_id_token(token)
    except:
        raise HTTPException(status_code=401, detail="Sesión inválida")

class BarrierInput(BaseModel):
    text: str

@app.post("/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(verify_token)):
    model = GenerativeModel(MODEL_NAME)
    # El prompt se basa en los objetivos de incidencia del proyecto [cite: 32, 34]
    prompt = f"Actúa como asesor del MNPT Costa Rica. Analiza esta barrera: {data.text}. Propón estrategias de incidencia política."
    response = model.generate_content(prompt)
    return {"analysis": response.text}

@app.get("/recommendations")
async def get_data(user=Depends(verify_token)):
    # Ruta estructurada según requerimiento
    docs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
    return [doc.to_dict() for doc in docs]
