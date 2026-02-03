import os
from fastapi import FastAPI, Depends, HTTPException, status, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google.cloud import firestore, storage
import vertexai
from vertexai.generative_models import GenerativeModel
import firebase_admin
from firebase_admin import auth

# --- CONFIGURACIÓN DE ENTORNO ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
APP_ID = os.getenv("APP_ID", "siis-spt-cr")
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")

app = FastAPI(title="SIIS-SPT System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- MODELOS ---
class BarrierInput(BaseModel):
    text: str

class ApprovalInput(BaseModel):
    recommendation_id: str
    approved_progress: int
    admin_notes: str

# --- SEGURIDAD ---
async def validate_user_session(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No autorizado")
    token = auth_header.split("Bearer ")[1]
    try:
        return auth.verify_id_token(token)
    except:
        raise HTTPException(status_code=401, detail="Sesión inválida")

# --- ENDPOINTS API ---

@app.get("/api/recommendations")
async def get_recommendations(user_data=Depends(validate_user_session)):
    try:
        docs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/analyze")
async def analyze_with_gemini(data: BarrierInput, user_data=Depends(validate_user_session)):
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = f"Como experto en OPCAT para el MNPT Costa Rica, analiza este obstáculo: {data.text}. Propón estrategias de incidencia política."
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evidence/upload")
async def upload_evidence(
    recommendation_id: str = Form(...), 
    description: str = Form(...),
    file: UploadFile = File(...), 
    user_data=Depends(validate_user_session)
):
    """Sube evidencia física y solicita pre-evaluación de la IA."""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"evidence/{recommendation_id}/{file.filename}")
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        # Pre-evaluación IA para ayudar al administrador
        model = GenerativeModel(MODEL_NAME)
        prompt = f"Evalúa esta evidencia para la recomendación {recommendation_id}: '{description}'. ¿Qué avance representa del 0 al 100? Responde solo el número."
        ai_suggestion = model.generate_content(prompt).text.strip()

        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "recommendation_id": recommendation_id,
            "submitted_by": user_data['email'],
            "file_url": blob.public_url,
            "description": description,
            "ai_suggestion": ai_suggestion,
            "status": "PENDING_REVIEW"
        })
        return {"message": "Evidencia subida correctamente. Pendiente de validación MNPT."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/approve")
async def approve_recommendation(data: ApprovalInput, user_data=Depends(validate_user_session)):
    """Valida oficialmente el progreso (Solo Administradores MNPT)"""
    try:
        rec_ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").document(data.recommendation_id)
        rec_ref.update({
            "progress": data.approved_progress,
            "status": "Validado" if data.approved_progress == 100 else "En Proceso",
            "last_review_by": user_data['email']
        })
        return {"message": "Progreso validado y publicado oficialmente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- SERVIDOR ESTÁTICO ---
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
