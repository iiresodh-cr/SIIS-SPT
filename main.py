import os
import io
import logging
import datetime
import json
import re
import google.auth
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

# --- CONFIGURACIÓN DE AUDITORÍA ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIIS-SPT-SERVER")

# --- VARIABLES DE ENTORNO ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = "siis-spt-cr"
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
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

# --- MODELOS DE DATOS ---
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
            raise HTTPException(status_code=403, detail="Usuario no registrado en Firestore")
            
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
        # Capturamos Firestore ID y el campo 'id' interno para protagonismo
        all_recs = [{"firestore_doc_id": doc.id, **doc.to_dict()} for doc in docs]
        
        if user["is_admin"]:
            return all_recs
        
        slug = user.get("inst_slug", "").lower()
        return [r for r in all_recs if slug in r.get("institution", "").lower()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/suggest-progress")
async def suggest_progress(data: SuggestionInput, user=Depends(get_current_user)):
    if not user["is_admin"]:
        raise HTTPException(status_code=403)
    try:
        rec_doc = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").document(data.recommendation_id).get()
        sub_doc = db.collection("artifacts").document(APP_ID).collection("submissions").document(data.submission_id).get()
        
        meta = rec_doc.to_dict().get("description", "")
        logro = sub_doc.to_dict().get("description", "")

        model = GenerativeModel(MODEL_NAME)
        prompt = f"Analiza cumplimiento. Meta: '{meta}'. Logro: '{logro}'. Responde JSON: {{'percentage': int, 'justification': str}}"
        response = model.generate_content(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        return json.loads(match.group())
    except Exception as e:
        logger.error(f"Fallo IA PIDA: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en el motor")

@app.post("/api/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(get_current_user)):
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = f"PIDA: Analiza barrera institucional para el MNPT: {data.text}"
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evidence/upload")
async def upload_evidence(
    recommendation_id: str = Form(...), 
    description: str = Form(...),
    file: UploadFile = File(...), 
    user=Depends(get_current_user)
):
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        file_path = f"evidence/{recommendation_id}/{file.filename}"
        blob = bucket.blob(file_path)
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "id": sub_ref.id,
            "recommendation_id": recommendation_id, 
            "submitted_by": user["email"],
            "description": description,
            "file_path": file_path,
            "status": "PENDIENTE",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Registro completado"}
    except Exception as e:
        logger.error(f"Fallo carga: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/pending")
async def list_pending(user=Depends(get_current_user)):
    """Solución 403: Firma URLs con dominio storage.googleapis.com."""
    if not user["is_admin"]:
        raise HTTPException(status_code=403)
    try:
        creds, _ = google.auth.default()
        subs_stream = db.collection("artifacts").document(APP_ID).collection("submissions").where("status", "==", "PENDIENTE").stream()
        bucket = storage_client.bucket(BUCKET_NAME)
        results = []
        
        for s in subs_stream:
            data = s.to_dict()
            file_path = data.get("file_path")
            
            if file_path:
                blob = bucket.blob(file_path)
                # Generación V4 con dominio oficial para evitar Forbidden
                data["file_url"] = blob.generate_signed_url(
                    version="v4", 
                    expiration=datetime.timedelta(minutes=60), 
                    method="GET",
                    service_account_email=creds.service_account_email
                )
            
            # Recuperamos campo 'id' interno para protagonismo visual
            rec_doc = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").document(data['recommendation_id']).get()
            if rec_doc.exists:
                rec_info = rec_doc.to_dict()
                data["display_id"] = rec_info.get("id", "S/N")
                data["current_progress"] = rec_info.get("progress", 0)
            
            if "timestamp" in data and data["timestamp"]:
                data["timestamp"] = data["timestamp"].isoformat()
            results.append(data)
            
        return results
    except Exception as e:
        logger.error(f"Error pendientes: {str(e)}")
        raise HTTPException(status_code=500, detail="Error de servidor")

@app.post("/api/admin/approve")
async def approve_submission(action: SubmissionAction, user=Depends(get_current_user)):
    if not user["is_admin"]:
        raise HTTPException(status_code=403)
    try:
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document(action.submission_id)
        sub_doc = sub_ref.get()
        rec_id = sub_doc.to_dict().get("recommendation_id")
        
        sub_ref.update({"status": "APROBADO"})
        rec_ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").document(rec_id)
        rec_ref.update({
            "progress": action.progress,
            "status": "Completado" if action.progress >= 100 else "En Progreso",
            "last_validated": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Actualización exitosa"}
    except Exception as e:
        logger.error(f"Fallo aprobación: {str(e)}")
        raise HTTPException(status_code=500, detail="Fallo de base de datos")

@app.get("/api/report/generate")
async def generate_pdf(user=Depends(get_current_user)):
    recs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 750, "INFORME TÉCNICO SPT")
    y = 700
    for doc in recs:
        d = doc.to_dict()
        p.setFont("Helvetica-Bold", 10)
        p.drawString(100, y, f"[{d.get('id')}] {d.get('institution')} - {d.get('progress')}%")
        y -= 30
        if y < 100: p.showPage(); y = 750
    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
