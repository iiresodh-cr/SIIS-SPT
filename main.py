import os
import io
from typing import List, Dict
from fastapi import FastAPI, Depends, HTTPException, status, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from google.cloud import firestore, storage
import vertexai
from vertexai.generative_models import GenerativeModel
import firebase_admin
from firebase_admin import auth
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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

# Inicialización de servicios
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

# --- SEGURIDAD (CORREGIDA) ---
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
    """Obtiene las observaciones/recomendaciones base de la línea de base."""
    try:
        docs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/analyze")
async def analyze_with_gemini(data: BarrierInput, user_data=Depends(validate_user_session)):
    """Laboratorio de IA para análisis de barreras."""
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = f"Analiza esta barrera institucional para el MNPT Costa Rica: {data.text}. Propón 3 estrategias de incidencia política."
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
    """Carga de pruebas por el usuario e inferencia preliminar de la IA."""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"evidence/{recommendation_id}/{file.filename}")
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        # IA pre-evalúa el nivel de cumplimiento basado en la descripción
        model = GenerativeModel(MODEL_NAME)
        prompt = f"La institución envió esta prueba: '{description}'. ¿Qué porcentaje de avance (0-100) sugiere esto para la recomendación {recommendation_id}? Responde solo el número."
        ai_suggestion = model.generate_content(prompt).text.strip()

        # Registro de la evidencia para validación del MNPT
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "recommendation_id": recommendation_id,
            "submitted_by": user_data['email'],
            "file_url": blob.public_url,
            "description": description,
            "ai_suggestion": ai_suggestion,
            "status": "PENDING_REVIEW",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Evidencia subida. Pendiente de validación por el administrador del MNPT."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/generate")
async def generate_pdf_report(user_data=Depends(validate_user_session)):
    """Genera el informe oficial PDF para el SPT."""
    try:
        docs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.setTitle("Informe SIIS-SPT")
        
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 750, "Informe de Seguimiento SIIS-SPT Costa Rica")
        p.setFont("Helvetica", 10)
        p.drawString(100, 730, f"Generado para: {user_data['email']}")
        
        y = 700
        for doc in docs:
            d = doc.to_dict()
            p.setFont("Helvetica-Bold", 11)
            p.drawString(100, y, f"[{d.get('id', 'N/A')}] {d.get('institution', 'N/A')}")
            y -= 15
            p.setFont("Helvetica", 10)
            p.drawString(100, y, f"Estado: {d.get('status', 'N/A')} | Progreso: {d.get('progress', 0)}%")
            y -= 30
            if y < 100:
                p.showPage()
                y = 750
        p.save()
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": "attachment;filename=Reporte_SIIS_SPT.pdf"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Servir Frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
