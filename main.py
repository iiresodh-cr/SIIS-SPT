import os
import io
from typing import Optional, List
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

# --- CONFIGURACIÓN E INYECCIÓN DE VARIABLES ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = "siis-spt-cr"
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")
# Selección de modelo por variable de entorno obligatoria
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
LOCATION = "us-central1"

app = FastAPI(
    title="SIIS-SPT Sistema Integral de Auditoría",
    description="Plataforma de monitoreo de recomendaciones internacionales para Costa Rica"
)

# Configuración de seguridad CORS para el Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización de Firebase Admin y Google Cloud Services
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- MODELOS DE DATOS PARA VALIDACIÓN DE ENTRADA ---
class BarrierInput(BaseModel):
    text: str

class ValidationInput(BaseModel):
    recommendation_id: str
    submission_id: str
    approved_progress: int
    notes: Optional[str] = ""

# --- MIDDLEWARE DE SEGURIDAD (RBAC DESDE FIRESTORE) ---
async def get_authorized_user(request: Request):
    """
    Cruza el token de Firebase Auth con la colección 'users' para obtener 
    permisos de Administrador o Institución.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Es necesario un token de sesión válido")
    
    token = auth_header.split("Bearer ")[1]
    try:
        # 1. Validar la identidad del usuario en Firebase
        decoded_token = auth.verify_id_token(token)
        email = decoded_token.get("email")
        
        # 2. Consultar la colección de usuarios creada en Firestore
        user_ref = db.collection("artifacts").document(APP_ID).collection("users").document(email)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(status_code=403, detail="Usuario no registrado en el sistema de permisos")
            
        user_data = user_doc.to_dict()
        # Verificación lógica de Administrador (IIRESODH) o acceso total
        user_data["is_admin"] = user_data.get("role") == "admin" or user_data.get("inst_slug") == "all"
        return user_data
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Falla de autorización: {str(e)}")

# --- ENDPOINTS DE RECOMENDACIONES ---

@app.get("/api/recommendations")
async def list_recommendations(user=Depends(get_authorized_user)):
    """
    Entrega el listado de recomendaciones aplicando el filtro institucional 
    para evitar la contaminación de datos.
    """
    try:
        ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
        docs = ref.stream()
        all_recs = [doc.to_dict() for doc in docs]
        
        # Si eres Administrador/IIRESODH con inst_slug 'all', no hay filtros
        if user["is_admin"]:
            return all_recs
        
        # Si eres una institución, solo ves lo tuyo
        slug = user.get("inst_slug", "").lower()
        return [r for r in all_recs if slug in r.get("institution", "").lower()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINTS DE GESTIÓN DE EVIDENCIA ---

@app.post("/api/evidence/upload")
async def upload_evidence(
    recommendation_id: str = Form(...), 
    description: str = Form(...),
    file: UploadFile = File(...), 
    user=Depends(get_authorized_user)
):
    """
    Carga pruebas físicas a Storage y crea un registro de 'submission' para ser validado.
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        file_path = f"evidence/{recommendation_id}/{file.filename}"
        blob = bucket.blob(file_path)
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        # Crear entrada en la cola de revisión administrativa
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "id": sub_ref.id,
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "institution": user.get("institution", "IIRESODH"),
            "description": description,
            "file_url": blob.public_url,
            "status": "PENDIENTE",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "La evidencia ha sido enviada para validación del Administrador"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/pending")
async def list_pending_submissions(user=Depends(get_authorized_user)):
    """Solo el administrador puede ver y gestionar las validaciones."""
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Permisos insuficientes para administrar evidencias")
    
    subs = db.collection("artifacts").document(APP_ID).collection("submissions").where("status", "==", "PENDIENTE").stream()
    return [s.to_dict() for s in subs]

# --- ENDPOINTS DE INTELIGENCIA E INFORMES ---

@app.post("/api/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(get_authorized_user)):
    """
    Laboratorio de Incidencia Política utilizando el modelo gemini-2.5-flash 
    configurado en variables de entorno.
    """
    try:
        model = GenerativeModel(MODEL_NAME)
        prompt = (
            f"Actúa como consultor senior del Subcomité para la Prevención de la Tortura (SPT). "
            f"Analiza técnicamente el siguiente obstáculo reportado en Costa Rica: '{data.text}'. "
            f"Propón una hoja de ruta de 3 pasos para superarlo mediante incidencia política."
        )
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Vertex AI: {str(e)}")

@app.get("/api/report/generate")
async def generate_pdf_report(user=Depends(get_authorized_user)):
    """
    Genera el informe oficial consolidado del estado de las 29 recomendaciones.
    """
    try:
        docs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Estilo del PDF
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, 750, "SIIS-SPT: Informe Oficial de Cumplimiento")
        p.setFont("Helvetica", 10)
        p.drawString(50, 730, f"Generado por Administrador: {user['email']} (IIRESODH)")
        p.line(50, 720, 550, 720)
        
        y = 690
        for doc in docs:
            d = doc.to_dict()
            p.setFont("Helvetica-Bold", 11)
            p.drawString(50, y, f"[{d.get('id', 'N/A')}] {d.get('institution', 'N/A')}")
            y -= 15
            p.setFont("Helvetica", 10)
            p.drawString(60, y, f"Estado: {d.get('status', 'Pendiente')} | Progreso Validado: {d.get('progress', 0)}%")
            y -= 35
            
            # Control de salto de página
            if y < 80:
                p.showPage()
                y = 750
        
        p.save()
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf", headers={
            "Content-Disposition": "attachment; filename=Informe_SIIS_SPT.pdf"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- MONTAJE DEL SERVIDOR ESTÁTICO ---
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Cloud Run inyecta el puerto dinámicamente
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
