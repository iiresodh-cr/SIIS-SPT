import os
import io
import logging
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

# --- CONFIGURACIÓN DE LOGGING PARA AUDITORÍA ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIIS-SPT-BACKEND")

# --- VARIABLES DE ENTORNO Y CONFIGURACIÓN ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = "siis-spt-cr"  # ID de la aplicación en Firestore
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")
# Modelo definido por variable de entorno según tu orden: gemini-2.5-flash
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
LOCATION = "us-central1"

# --- INICIALIZACIÓN DE SERVICIOS DE GOOGLE CLOUD ---
app = FastAPI(
    title="SIIS-SPT: Sistema Integral de Monitoreo",
    description="Plataforma técnica para el seguimiento de recomendaciones CAT y SPT en Costa Rica",
    version="1.0.0"
)

# Configuración robusta de CORS para entornos de desarrollo y producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Inicialización segura de Firebase Admin
if not firebase_admin._apps:
    try:
        firebase_admin.initialize_app()
        logger.info("Firebase Admin inicializado correctamente.")
    except Exception as e:
        logger.error(f"Error al inicializar Firebase: {e}")

# Clientes de servicios
db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- MODELOS DE DATOS PANTIC ---
class BarrierInput(BaseModel):
    text: str

class ValidationAction(BaseModel):
    recommendation_id: str
    submission_id: str
    approved_progress: int
    admin_notes: str

# --- LÓGICA DE AUTORIZACIÓN Y ROLES (RBAC) ---
async def get_current_active_user(request: Request) -> Dict[str, Any]:
    """
    Middleware de seguridad que valida el token y recupera el perfil de Firestore.
    Cruza los datos con la colección /users/.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Se requiere autenticación Bearer Token"
        )
    
    token = auth_header.split("Bearer ")[1]
    try:
        # Verificación del token con Firebase Auth
        decoded_token = auth.verify_id_token(token)
        email = decoded_token.get("email")
        
        # Consulta al documento del usuario en Firestore
        user_ref = db.collection("artifacts").document(APP_ID).collection("users").document(email)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            logger.warning(f"Intento de acceso de usuario no registrado: {email}")
            raise HTTPException(status_code=403, detail="Usuario no registrado en el padrón del sistema")
            
        user_data = user_doc.to_dict()
        # Lógica de Administrador: IIRESODH o rol admin
        user_data["is_admin"] = user_data.get("role") == "admin" or user_data.get("inst_slug") == "all"
        return user_data
        
    except Exception as e:
        logger.error(f"Falla en verificación de token: {str(e)}")
        raise HTTPException(status_code=401, detail="Sesión expirada o inválida")

# --- ENDPOINTS DE RECOMENDACIONES ---

@app.get("/api/recommendations")
async def get_recommendations(user=Depends(get_current_active_user)):
    """
    Obtiene las recomendaciones aplicando el aislamiento de datos institucional.
    Busca en la ruta: /artifacts/siis-spt-cr/public/data/recommendations/
    """
    try:
        # Referencia a la colección principal
        recs_ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
        docs = recs_ref.stream()
        
        recommendations_list = []
        for doc in docs:
            data = doc.to_dict()
            # Asegurar que el ID del documento esté presente en el diccionario
            if "id" not in data:
                data["id"] = doc.id
            recommendations_list.append(data)
            
        # Si el usuario es ADMIN (IIRESODH), ve todas las 29+ recomendaciones
        if user.get("is_admin"):
            return recommendations_list
            
        # Si es usuario institucional, filtrar por inst_slug
        slug = user.get("inst_slug", "").lower()
        if not slug:
            return [] # No tiene permisos de visualización
            
        return [r for r in recommendations_list if slug in r.get("institution", "").lower()]
        
    except Exception as e:
        logger.error(f"Error al listar recomendaciones: {e}")
        raise HTTPException(status_code=500, detail="Error al consultar Firestore")

# --- ENDPOINTS DE GESTIÓN DE EVIDENCIAS ---

@app.post("/api/evidence/upload")
async def upload_evidence_file(
    recommendation_id: str = Form(...),
    description: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(get_current_active_user)
):
    """
    Sube archivos PDF a Cloud Storage y crea un registro de auditoría en 'submissions'.
    """
    try:
        # 1. Subida al Bucket de Storage
        bucket = storage_client.bucket(BUCKET_NAME)
        file_name = f"evidence/{recommendation_id}/{file.filename}"
        blob = bucket.blob(file_name)
        
        blob.upload_from_file(file.file, content_type=file.content_type)
        logger.info(f"Archivo {file.filename} subido a Storage por {user['email']}")
        
        # 2. Registro de la entrega en Firestore para validación administrativa
        submission_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        submission_data = {
            "id": submission_ref.id,
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "institution": user.get("institution", "N/A"),
            "description": description,
            "file_url": blob.public_url,
            "status": "PENDIENTE",
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        submission_ref.set(submission_data)
        
        return {"message": "Evidencia cargada exitosamente. Pendiente de validación MNPT/IIRESODH."}
        
    except Exception as e:
        logger.error(f"Error en carga de evidencia: {e}")
        raise HTTPException(status_code=500, detail="Falla en el proceso de carga")

@app.get("/api/admin/pending")
async def get_pending_submissions(user=Depends(get_current_active_user)):
    """Solo accesible para administradores (IIRESODH/MNPT)."""
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Permisos administrativos requeridos")
        
    try:
        subs_ref = db.collection("artifacts").document(APP_ID).collection("submissions")
        pending = subs_ref.where("status", "==", "PENDIENTE").stream()
        return [s.to_dict() for s in pending]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINTS DE INTELIGENCIA ARTIFICIAL ---

@app.post("/api/ai/analyze")
async def analyze_institutional_barrier(data: BarrierInput, user=Depends(get_current_active_user)):
    """
    Laboratorio de Incidencia: Utiliza el modelo gemini-2.5-flash definido por el usuario.
    """
    try:
        # Instanciar el modelo dinámicamente desde la variable de entorno
        model = GenerativeModel(MODEL_NAME)
        
        prompt = (
            f"Como asesor técnico experto del Subcomité para la Prevención de la Tortura (SPT) y del MNPT, "
            f"analiza la siguiente barrera institucional reportada en el SIIS-SPT: '{data.text}'. "
            f"Genera una estrategia de incidencia política de 3 niveles para superar este obstáculo."
        )
        
        response = model.generate_content(prompt)
        return {"analysis": response.text}
        
    except Exception as e:
        logger.error(f"Falla en Vertex AI ({MODEL_NAME}): {e}")
        raise HTTPException(status_code=500, detail=f"Error en motor de IA: {str(e)}")

# --- ENDPOINT DE REPORTES (PDF) ---

@app.get("/api/report/generate")
async def generate_official_pdf_report(user=Depends(get_current_active_user)):
    """
    Genera el informe oficial consolidado de las 29+ recomendaciones.
    """
    try:
        recs_ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
        docs = recs_ref.stream()
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Diseño del Informe
        p.setFillColor(colors.HexColor("#1e293b")) # Slate 800
        p.rect(0, height-1.2*inch, width, 1.2*inch, fill=1)
        
        p.setFont("Helvetica-Bold", 18)
        p.setFillColor(colors.white)
        p.drawString(0.5*inch, height-0.6*inch, "SIIS-SPT: Informe de Monitoreo Costa Rica")
        
        p.setFont("Helvetica", 10)
        p.drawString(0.5*inch, height-0.9*inch, f"Generado por: {user['email']} | Institución: {user['institution']}")
        
        y = height - 1.8*inch
        p.setFillColor(colors.black)
        
        for doc in docs:
            d = doc.to_dict()
            p.setFont("Helvetica-Bold", 12)
            p.drawString(0.5*inch, y, f"[{d.get('id', doc.id)}] {d.get('institution', 'N/A')}")
            y -= 0.2*inch
            
            p.setFont("Helvetica", 10)
            status_text = f"Avance: {d.get('progress', 0)}% | Estado: {d.get('status', 'Pendiente')}"
            p.drawString(0.6*inch, y, status_text)
            
            y -= 0.5*inch
            if y < 1*inch:
                p.showPage()
                y = height - 1*inch
                
        p.save()
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Reporte_SIIS_SPT_CRI.pdf"}
        )
        
    except Exception as e:
        logger.error(f"Error al generar PDF: {e}")
        raise HTTPException(status_code=500, detail="Falla en la generación del documento")

# --- SERVIDOR DE ARCHIVOS ESTÁTICOS ---
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_home_page():
    return FileResponse(os.path.join("static", "index.html"))

# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    import uvicorn
    # Se utiliza el puerto inyectado por Cloud Run
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
