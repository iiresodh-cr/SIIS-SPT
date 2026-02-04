import os
import io
import logging
import datetime
import json
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, Form, status
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
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- CONFIGURACIÓN DE AUDITORÍA ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIIS-SPT-SERVER")

# --- VARIABLES DE ENTORNO ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = "siis-spt-cr"
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash") #
LOCATION = "us-central1"

app = FastAPI(title="SIIS-SPT PRO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- MODELOS ---
class BarrierInput(BaseModel):
    text: str

class SubmissionAction(BaseModel):
    submission_id: str
    progress: int

class SuggestionInput(BaseModel):
    submission_id: str
    recommendation_id: str

# --- MIDDLEWARE DE SEGURIDAD ---
async def get_current_user(request: Request) -> Dict[str, Any]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No autorizado")
    
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        email = decoded_token.get("email")
        
        user_ref = db.collection("artifacts").document(APP_ID).collection("users").document(email)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(status_code=403, detail="Usuario no registrado")
            
        user_data = user_doc.to_dict()
        user_data["is_admin"] = user_data.get("role") == "admin" or user_data.get("inst_slug") == "all"
        return user_data
    except Exception as e:
        logger.error(f"Error de Auth: {str(e)}")
        raise HTTPException(status_code=401, detail="Token inválido")

# --- API ENDPOINTS ---

@app.get("/api/auth/me")
async def auth_me(user=Depends(get_current_user)):
    return {
        "email": user.get("email"),
        "institution": user.get("institution"),
        "is_admin": user.get("is_admin"),
        "role": user.get("role")
    }

@app.get("/api/recommendations")
async def list_recommendations(user=Depends(get_current_user)):
    try:
        ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
        docs = ref.stream()
        all_recs = [{"firestore_id": doc.id, **doc.to_dict()} for doc in docs]
        
        if user["is_admin"]:
            return all_recs
        
        slug = user.get("inst_slug", "").lower()
        return [r for r in all_recs if slug in r.get("institution", "").lower()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/suggest-progress")
async def suggest_progress(data: SuggestionInput, user=Depends(get_current_user)):
    """
    IA Consultora PIDA: Compara la meta con la evidencia y propone un % de avance.
    """
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Solo administradores")
    
    try:
        # 1. Obtener la meta (Recomendación)
        rec_ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").document(data.recommendation_id)
        rec_doc = rec_ref.get()
        
        # 2. Obtener la evidencia (Submission)
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document(data.submission_id)
        sub_doc = sub_ref.get()
        
        if not rec_doc.exists or not sub_doc.exists:
            raise HTTPException(status_code=404, detail="Datos no encontrados")

        meta = rec_doc.to_dict().get("description", "")
        logro = sub_doc.to_dict().get("description", "")

        model = GenerativeModel(MODEL_NAME)
        prompt = f"""
        Como auditor senior del MNPT, evalúa este avance técnico.
        RECOMENDACIÓN ORIGINAL: "{meta}"
        LOGRO REPORTADO: "{logro}"

        Determina qué porcentaje de cumplimiento (0 a 100) representa este logro respecto a la meta total. 
        Responde exclusivamente en formato JSON con dos campos:
        "percentage": (un número entero)
        "justification": (una breve explicación técnica de por qué ese valor)
        """
        
        response = model.generate_content(prompt)
        # Limpieza básica de la respuesta para extraer JSON
        clean_json = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        return json.loads(clean_json)
    except Exception as e:
        logger.error(f"Error en Sugerencia IA: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al generar propuesta de IA")

@app.post("/api/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(get_current_user)):
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = f"""
        Actúa como la inteligencia central de PIDA (Plataforma de Investigación y Defensa Avanzada). 
        Analiza este obstáculo para el MNPT de Costa Rica: "{data.text}"
        Genera: 1. DIAGNÓSTICO, 2. ESTRATEGIA DE DEFENSA, 3. TÁCTICAS DE INCIDENCIA.
        Tono ejecutivo y profesional.
        """
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error en IA.")

@app.post("/api/evidence/upload")
async def upload_evidence(
    recommendation_id: str = Form(...), 
    description: str = Form(...),
    file: UploadFile = File(...), 
    user=Depends(get_current_user)
):
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"evidence/{recommendation_id}/{file.filename}")
        blob.upload_from_file(file.file, content_type=file.content_type)
        signed_url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=10080), method="GET")
        
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "id": sub_ref.id,
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "description": description,
            "file_url": signed_url,
            "status": "PENDIENTE",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Subida exitosa"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/pending")
async def list_pending(user=Depends(get_current_user)):
    if not user["is_admin"]: raise HTTPException(status_code=403)
    subs = db.collection("artifacts").document(APP_ID).collection("submissions").where("status", "==", "PENDIENTE").stream()
    results = []
    for s in subs:
        d = s.to_dict()
        if "timestamp" in d and d["timestamp"]: d["timestamp"] = d["timestamp"].isoformat()
        results.append(d)
    return results

@app.post("/api/admin/approve")
async def approve_submission(action: SubmissionAction, user=Depends(get_current_user)):
    if not user["is_admin"]: raise HTTPException(status_code=403)
    sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document(action.submission_id)
    sub_doc = sub_ref.get()
    if not sub_doc.exists: raise HTTPException(status_code=404)
    
    rec_id = sub_doc.to_dict().get("recommendation_id")
    sub_ref.update({"status": "APROBADO"})
    
    rec_ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").document(rec_id)
    rec_ref.update({
        "progress": action.progress,
        "status": "Completado" if action.progress >= 100 else "En Progreso",
        "last_validated": firestore.SERVER_TIMESTAMP
    })
    return {"message": "Actualizado"}

@app.get("/api/report/generate")
async def generate_pdf(user=Depends(get_current_user)):
    recs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16); p.drawString(50, 750, "SIIS-SPT: Informe"); y = 700
    for doc in recs:
        d = doc.to_dict()
        p.setFont("Helvetica-Bold", 10); p.drawString(50, y, f"[{d.get('id')}] {d.get('institution')}")
        p.setFont("Helvetica", 10); p.drawString(50, y-15, f"Avance: {d.get('progress', 0)}%"); y -= 45
        if y < 100: p.showPage(); y = 750
    p.save(); buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index(): return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
