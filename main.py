import os
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel
import firebase_admin
from firebase_admin import auth

# --- CONFIGURACIÓN ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
APP_ID = os.getenv("APP_ID", "siis-spt-cr")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INICIALIZACIÓN ---
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- MODELOS ---
class BarrierInput(BaseModel):
    text: str

# --- SEGURIDAD CORREGIDA ---
# Eliminamos tipos complejos en la firma para evitar fallos de inspección
async def get_current_user(authorization: str = Security(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Header de autorización inválido"
        )
    
    token = authorization.split("Bearer ")[1]
    try:
        # Esto devuelve un diccionario con los datos del usuario
        return auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Error de token: {str(e)}"
        )

# --- ENDPOINTS ---

@app.get("/recommendations")
async def list_recommendations(user=Depends(get_current_user)):
    """
    Lista el estado de cumplimiento del 100% de las recomendaciones[cite: 37].
    """
    try:
        # Referencia exacta: /artifacts/siis-spt-cr/public/data/recommendations
        docs = db.collection("artifacts") \
                 .document(APP_ID) \
                 .collection("public") \
                 .document("data") \
                 .collection("recommendations").stream()
        
        results = []
        for doc in docs:
            results.append(doc.to_dict())
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(get_current_user)):
    """
    Usa IA para generar estrategias contra la falta de voluntad política 
    o recursos[cite: 19, 20].
    """
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = (
            f"Contexto: Proyecto SIIS-SPT Costa Rica para el MNPT[cite: 3, 7]. "
            f"Problema: {data.text}. "
            f"Tarea: Genera una estrategia de incidencia política basada en evidencia "
            f"para superar esta barrera institucional[cite: 32]."
        )
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "SIIS-SPT API activa"}
