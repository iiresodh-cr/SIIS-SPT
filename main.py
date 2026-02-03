import os
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

app = FastAPI(title="SIIS-SPT System")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INICIALIZACIÓN DE SERVICIOS ---
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- MODELOS DE DATOS ---
class BarrierInput(BaseModel):
    text: str

# --- SEGURIDAD (CORREGIDA PARA EVITAR VALUEERROR) ---
async def validate_user_session(request: Request):
    """
    Valida el token de Firebase extrayéndolo directamente del request
    para evitar errores de inspección de firmas en FastAPI.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        # Nota: Durante pruebas iniciales puedes comentar el raise si no tienes el JWT listo
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Se requiere autenticación de Firebase"
        )
    
    token = auth_header.split("Bearer ")[1]
    try:
        return auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Sesión inválida: {str(e)}"
        )

# --- ENDPOINTS DE API ---

@app.get("/api/recommendations")
async def get_recommendations(user_data=Depends(validate_user_session)):
    """
    Consulta Firestore para obtener el cumplimiento de las recomendaciones.
    Ruta: /artifacts/{APP_ID}/public/data/recommendations
    """
    try:
        collection_path = f"artifacts/{APP_ID}/public/data/recommendations"
        docs = db.collection(collection_path).stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/analyze")
async def analyze_with_gemini(data: BarrierInput, user_data=Depends(validate_user_session)):
    """
    Laboratorio de IA: Analiza obstáculos institucionales del MNPT.
    """
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = (
            f"Contexto: Sistema SIIS-SPT (Costa Rica). "
            f"Analiza el siguiente obstáculo institucional para el MNPT: {data.text}. "
            f"Genera 3 estrategias de incidencia política y técnica."
        )
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Gemini: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "online", "service": "SIIS-SPT"}

# --- SERVIDOR DE ARCHIVOS ESTÁTICOS (INTERFAZ WEB) ---

# 1. Montamos la carpeta 'static' para que JS/CSS sean accesibles
# Si no tienes carpeta static aún, crea una y mete el index.html ahí.
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    """
    Sirve el archivo index.html en la raíz de la URL.
    """
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Archivo index.html no encontrado en la carpeta /static"}

# --- INICIO ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
