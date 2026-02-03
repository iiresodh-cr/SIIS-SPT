import os
import io
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, Form
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

# --- CONFIGURACIÓN E ID DEL PROYECTO ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = "siis-spt-cr"
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")
LOCATION = "us-central1"

app = FastAPI(title="SIIS-SPT Sistema de Auditoría")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización de Servicios
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- MODELOS DE DATOS ---
class BarrierInput(BaseModel):
    text: str

class ValidationInput(BaseModel):
    recommendation_id: str
    submission_id: str
    approved_progress: int
    notes: str

# --- DEPENDENCIAS DE SEGURIDAD (RBAC) ---
async def get_authorized_user(request: Request):
    """
    Verifica el token de Firebase y cruza los datos con la colección 'users' de Firestore.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token ausente o mal formado")
    
    token = auth_header.split("Bearer ")[1]
    try:
        # 1. Validar identidad en Firebase Auth
        decoded_token = auth.verify_id_token(token)
        email = decoded_token.get("email")
        
        # 2. Consultar permisos en la colección 'users' creada por el usuario
        user_ref = db.collection("artifacts").document(APP_ID).collection("users").document(email)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(status_code=403, detail="Usuario no autorizado en el padrón institucional")
            
        user_data = user_doc.to_dict()
        # Inyectar banderas de rol para el resto del código
        user_data["is_admin"] = user_data.get("role") == "admin" or user_data.get("inst_slug") == "all"
        return user_data
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Error de autenticación: {str(e)}")

# --- ENDPOINTS DE RECOMENDACIONES ---

@app.get("/api/recommendations")
async def list_recommendations(user=Depends(get_authorized_user)):
    """
    Lista las recomendaciones desde /public/data/recommendations.
    Implementa filtrado institucional para evitar contaminación de datos.
    """
    try:
        ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
        docs = ref.stream()
        all_recs = [doc.to_dict() for doc in docs]
        
        # Si es Admin o IIRESODH (inst_slug: 'all'), lo ve TODO
        if user["is_admin"]:
            return all_recs
        
        # Si es una institución específica, filtra para que no vea datos de otras
        slug = user.get("inst_slug", "").lower()
        return [r for r in all_recs if slug in r.get("institution", "").lower()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en base de datos: {str(e)}")

# --- ENDPOINTS DE EVIDENCIA Y ADMINISTRACIÓN ---

@app.post("/api/evidence/upload")
async def upload_evidence(
    recommendation_id: str = Form(...), 
    description: str = Form(...),
    file: UploadFile = File(...), 
    user=Depends(get_authorized_user)
):
    """
    Sube archivos PDF a Cloud Storage y crea un ticket de validación para el MNPT.
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        file_path = f"evidence/{recommendation_id}/{file.filename}"
        blob = bucket.blob(file_path)
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        # Crear entrada en la colección de 'submissions' para auditoría
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "id": sub_ref.id,
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "institution": user["institution"],
            "description": description,
            "file_url": blob.public_url,
            "status": "PENDIENTE",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Evidencia subida correctamente. El MNPT procederá con la validación."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/pending")
async def list_pending_submissions(user=Depends(get_authorized_user)):
    """Solo el Admin (IIRESODH/MNPT) puede ver la cola de validación."""
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Acceso denegado a funciones de administrador")
    
    subs = db.collection("artifacts").document(APP_ID).collection("submissions").where("status", "==", "PENDIENTE").stream()
    return [s.to_dict() for s in subs]

# --- ENDPOINTS DE INTELIGENCIA E INFORMES ---

@app.post("/api/ai/analyze")
async def analyze_with_gemini(data: BarrierInput, user=Depends(get_authorized_user)):
    """Usa Gemini Flash para generar estrategias de incidencia técnica."""
    try:
        model = GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Actúa como consultor senior del Subcomité para la Prevención de la Tortura (SPT). "
            f"Analiza la siguiente barrera institucional reportada en Costa Rica: '{data.text}'. "
            f"Genera 3 acciones concretas de incidencia política para superar este obstáculo."
        )
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/generate")
async def generate_pdf_report(user=Depends(get_authorized_user)):
    """Genera un reporte PDF con el estado oficial de las recomendaciones."""
    try:
        docs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, 750, "SIIS-SPT: Informe Consolidado de Cumplimiento")
        p.setFont("Helvetica", 10)
        p.drawString(50, 730, f"Generado por: {user['email']} ({user['institution']})")
        p.line(50, 720, 550, 720)
        
        y = 680
        for doc in docs:
            d = doc.to_dict()
            p.setFont("Helvetica-Bold", 12)
            p.drawString(50, y, f"[{d.get('id', 'N/A')}] {d.get('institution', 'N/A')}")
            y -= 15
            p.setFont("Helvetica", 10)
            p.drawString(60, y, f"Estado: {d.get('status', 'Pendiente')} | Progreso: {d.get('progress', 0)}%")
            y -= 35
            if y < 100: p.showPage(); y = 750
            
        p.save()
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- SERVIDOR DE ARCHIVOS ESTÁTICOS ---
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Se usa el puerto inyectado por Cloud Run o 8080 por defecto
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
