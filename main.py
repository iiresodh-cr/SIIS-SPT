import os
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel
import firebase_admin
from firebase_admin import auth

# --- CONFIGURACIÓN DE ENTORNO ---
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

# --- SEGURIDAD (MÉTODO ALTERNATIVO PARA EVITAR VALUEERROR) ---
async def validate_user_session(request: Request):
    """
    Se usa 'Request' directamente en lugar de 'Security(None)'. 
    Esto evita que FastAPI intente inspeccionar la firma de la clase 'str'.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta token de autorización"
        )
    
    token = auth_header.split("Bearer ")[1]
    try:
        # Retorna los datos decodificados del usuario de Firebase
        return auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Sesión expirada o inválida: {str(e)}"
        )

# --- ENDPOINTS ---

@app.get("/recommendations")
async def get_recommendations(user_data=Depends(validate_user_session)):
    """
    Lista las recomendaciones cargadas para el MNPT.
    """
    try:
        # Estructura: /artifacts/siis-spt-cr/public/data/recommendations
        collection_path = f"artifacts/{APP_ID}/public/data/recommendations"
        docs = db.collection(collection_path).stream()
        
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en base de datos: {str(e)}")

@app.post("/ai/analyze")
async def analyze_with_gemini(data: BarrierInput, user_data=Depends(validate_user_session)):
    """
    Analiza barreras como la falta de voluntad política o recursos usando IA[cite: 10, 11].
    """
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = (
            f"Actúa como un experto en el OPCAT para el SIIS-SPT[cite: 3]. "
            f"Analiza este obstáculo institucional: {data.text}[cite: 15, 18]. "
            f"Genera una estrategia de incidencia política para el MNPT de Costa Rica[cite: 32]."
        )
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Gemini: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "app": "SIIS-SPT"}
