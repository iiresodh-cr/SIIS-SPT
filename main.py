import os
from typing import Dict
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel
import firebase_admin
from firebase_admin import auth, credentials

# --- CONFIGURACIÓN DE ENTORNO ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
APP_ID = os.getenv("APP_ID", "siis-spt-cr")

app = FastAPI(title="SIIS-SPT Backend")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INICIALIZACIÓN DE SERVICIOS ---
# Inicializa Firebase sin argumentos si corre en Cloud Run (usa Service Account por defecto)
try:
    firebase_admin.initialize_app()
except ValueError:
    # Ya inicializado
    pass

db = firestore.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- MODELOS DE DATOS ---
class BarrierInput(BaseModel):
    text: str

# --- SEGURIDAD (CORREGIDA) ---
async def get_current_user(authorization: str = Security(None)) -> Dict:
    """
    Valida el token de Firebase. Se cambió el nombre y la lógica 
    para evitar el error de firma de FastAPI.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales de autenticación no proporcionadas",
        )
    
    token = authorization.split("Bearer ")[1]
    try:
        # Verifica el token con Firebase Auth
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=f"Token inválido: {str(e)}"
        )

# --- ENDPOINTS ---

@app.get("/recommendations")
async def list_recommendations(user: Dict = Depends(get_current_user)):
    """
    Obtiene el cumplimiento del 100% de las recomendaciones[cite: 37].
    Sigue la ruta: /artifacts/{APP_ID}/public/data/recommendations
    """
    try:
        docs = db.collection("artifacts")\
                 .document(APP_ID)\
                 .collection("public")\
                 .document("data")\
                 .collection("recommendations").stream()
        
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/analyze")
async def analyze_barrier(data: BarrierInput, user: Dict = Depends(get_current_user)):
    """
    Laboratorio de IA para superar barreras institucionales como 
    falta de voluntad política o recursos[cite: 10, 19, 20].
    """
    try:
        model = GenerativeModel(MODEL_NAME)
        
        # Prompt técnico basado en el mandato del SPT y el MNPT [cite: 9, 32]
        prompt = (
            f"Como experto en el Protocolo Facultativo (OPCAT) y consultor para el "
            f"Mecanismo Nacional de Prevención de la Tortura de Costa Rica, analiza: '{data.text}'. "
            f"Identifica si es una barrera de recursos, voluntad política o capacitación[cite: 18]. "
            f"Propón 3 pasos concretos de incidencia para que el MNPT logre la implementación "
            f"de las recomendaciones del SPT[cite: 12]."
        )
        
        response = model.generate_content(prompt)
        return {
            "analysis": response.text,
            "model_used": MODEL_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Vertex AI: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "online", "project": "SIIS-SPT"}

# --- INICIO DE SERVIDOR ---
if __name__ == "__main__":
    import uvicorn
    # Cloud Run inyecta la variable PORT
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
