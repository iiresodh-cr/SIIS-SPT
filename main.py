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
        
        # Consulta de perfil en Firestore
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
    """IA PIDA: Analiza el gap entre recomendación y evidencia."""
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Solo administradores")
    
    try:
        # Obtener Meta y Logro
        rec_doc = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").document(data.recommendation_id).get()
        sub_doc = db.collection("artifacts").document(APP_ID).collection("submissions").document(data.submission_id).get()
        
        if not rec_doc.exists or not sub_doc.exists:
            raise HTTPException(status_code=404, detail="Documentos no encontrados")

        meta = rec_doc.to_dict().get("description", "")
        logro = sub_doc.to_dict().get("description", "")

        model = GenerativeModel(MODEL_NAME)
        prompt = f"""
        Como experto auditor del MNPT, evalúa este avance:
        RECOMENDACIÓN: "{meta}"
        EVIDENCIA PRESENTADA: "{logro}"

        Propón un porcentaje de cumplimiento (0-100) y justifica técnicamente.
        Responde exclusivamente en JSON: {{"percentage": valor, "justification": "texto"}}
        """
        
        response = model.generate_content(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        return json.loads(match.group())
    except Exception as e:
        logger.error(f"Fallo Sugerencia IA: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en Motor PIDA")

@app.post("/api/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(get_current_user)):
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = f"""
        Actúa como la inteligencia de PIDA (Plataforma de Investigación y Defensa Avanzada). 
        Analiza este obstáculo institucional para el MNPT: "{data.text}"
        Genera un informe con: 1. INVESTIGACIÓN DE CAUSAS, 2. ESTRATEGIA DE DEFENSA, 3. TÁCTICAS DE INCIDENCIA.
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
        # 1. Subida al Bucket
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"evidence/{recommendation_id}/{file.filename}")
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        # 2. Generación de URL Segura (Con fallback si fallan los permisos IAM)
        try:
            file_url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=10080), method="GET")
        except Exception:
            logger.warning("Fallo al firmar URL. Usando URL de consola.")
            file_url = f"https://storage.cloud.google.com/{BUCKET_NAME}/evidence/{recommendation_id}/{file.filename}"
        
        # 3. Escritura en Firestore
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "id": sub_ref.id,
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "description": description,
            "file_url": file_url,
            "status": "PENDIENTE",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Evidencia registrada correctamente"}
    except Exception as e:
        logger.error(f"FALLO CRÍTICO EN CARGA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en servidor: {str(e)}")

@app.get("/api/admin/pending")
async def list_pending(user=Depends(get_current_user)):
    if not user["is_admin"]: raise HTTPException(status_code=403)
    try:
        subs = db.collection("artifacts").document(APP_ID).collection("submissions").where("status", "==", "PENDIENTE").stream()
        results = []
        for s in subs:
            d = s.to_dict()
            if "timestamp" in d and d["timestamp"]: d["timestamp"] = d["timestamp"].isoformat()
            results.append(d)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al recuperar pendientes")

@app.post("/api/admin/approve")
async def approve_submission(action: SubmissionAction, user=Depends(get_current_user)):
    if not user["is_admin"]: raise HTTPException(status_code=403)
    try:
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document(action.submission_id)
        sub_doc = sub_ref.get()
        if not sub_doc.exists: raise HTTPException(status_code=404)
        
        rec_id = sub_doc.to_dict().get("recommendation_id")
        
        # Actualización de estados en cadena
        sub_ref.update({"status": "APROBADO"})
        rec_ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").document(rec_id)
        rec_ref.update({
            "progress": action.progress,
            "status": "Completado" if action.progress >= 100 else "En Progreso",
            "last_validated": firestore.SERVER_TIMESTAMP
        })
        return {"message": f"Avance actualizado al {action.progress}%"}
    except Exception as e:
        logger.error(f"Error en aprobación: {str(e)}")
        raise HTTPException(status_code=500, detail="Fallo en base de datos")

@app.get("/api/report/generate")
async def generate_pdf(user=Depends(get_current_user)):
    recs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16); p.drawString(50, 750, "SIIS-SPT: Informe Oficial"); y = 700
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
