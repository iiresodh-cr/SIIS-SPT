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

# --- CONFIGURACIÓN DE AUDITORÍA ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIIS-SPT-SERVER")

# --- VARIABLES DE ENTORNO ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = "siis-spt-cr"
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")
# Modelo configurado por el usuario: gemini-2.5-flash
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

# --- MIDDLEWARE DE SEGURIDAD ---
async def get_current_user(request: Request) -> Dict[str, Any]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No autorizado")
    
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        email = decoded_token.get("email")
        
        # Consulta dinámica a la colección 'users'
        user_ref = db.collection("artifacts").document(APP_ID).collection("users").document(email)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(status_code=403, detail="Usuario no registrado en Firestore")
            
        user_data = user_doc.to_dict()
        # Unificamos la lógica: Admin si el rol es 'admin' o el slug es 'all'
        user_data["is_admin"] = user_data.get("role") == "admin" or user_data.get("inst_slug") == "all"
        return user_data
    except Exception as e:
        logger.error(f"Error de Auth: {str(e)}")
        raise HTTPException(status_code=401, detail="Token inválido")

# --- API ENDPOINTS ---

@app.get("/api/auth/me")
async def auth_me(user=Depends(get_current_user)):
    """Permite al frontend conocer el rol del usuario sin hardcodear emails."""
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
        all_recs = [doc.to_dict() for doc in docs]
        
        if user["is_admin"]:
            return all_recs
        
        slug = user.get("inst_slug", "").lower()
        return [r for r in all_recs if slug in r.get("institution", "").lower()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(get_current_user)):
    """Diagnóstico del Error 500: Verificamos disponibilidad del modelo y cuotas."""
    try:
        # Validamos que el modelo exista en la configuración
        if not MODEL_NAME:
            raise ValueError("La variable GEMINI_MODEL_NAME no está configurada")
            
        model = GenerativeModel(MODEL_NAME)
        prompt = f"Analiza esta barrera institucional para el MNPT: {data.text}. Propón 3 soluciones."
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        logger.error(f"FALLO CRÍTICO IA ({MODEL_NAME}): {str(e)}")
        # Si el modelo gemini-2.5-flash no está disponible aún, devolvemos un error descriptivo
        raise HTTPException(
            status_code=500, 
            detail=f"Error en Vertex AI con el modelo {MODEL_NAME}. Verifique disponibilidad en {LOCATION}."
        )

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
        
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "id": sub_ref.id,
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "description": description,
            "file_url": blob.public_url,
            "status": "PENDIENTE",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Evidencia subida correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/pending")
async def list_pending(user=Depends(get_current_user)):
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Solo administradores")
    subs = db.collection("artifacts").document(APP_ID).collection("submissions").where("status", "==", "PENDIENTE").stream()
    return [s.to_dict() for s in subs]

@app.get("/api/report/generate")
async def generate_pdf(user=Depends(get_current_user)):
    recs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 750, "SIIS-SPT: Informe de Cumplimiento")
    y = 700
    for doc in recs:
        d = doc.to_dict()
        p.setFont("Helvetica-Bold", 10)
        p.drawString(50, y, f"[{d.get('id')}] {d.get('institution')}")
        p.setFont("Helvetica", 10)
        p.drawString(50, y-15, f"Avance: {d.get('progress', 0)}%")
        y -= 45
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)import os
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

# --- CONFIGURACIÓN DE AUDITORÍA ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIIS-SPT-SERVER")

# --- VARIABLES DE ENTORNO ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = "siis-spt-cr"
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")
# Modelo configurado por el usuario: gemini-2.5-flash
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

# --- MIDDLEWARE DE SEGURIDAD ---
async def get_current_user(request: Request) -> Dict[str, Any]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No autorizado")
    
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        email = decoded_token.get("email")
        
        # Consulta dinámica a la colección 'users'
        user_ref = db.collection("artifacts").document(APP_ID).collection("users").document(email)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(status_code=403, detail="Usuario no registrado en Firestore")
            
        user_data = user_doc.to_dict()
        # Unificamos la lógica: Admin si el rol es 'admin' o el slug es 'all'
        user_data["is_admin"] = user_data.get("role") == "admin" or user_data.get("inst_slug") == "all"
        return user_data
    except Exception as e:
        logger.error(f"Error de Auth: {str(e)}")
        raise HTTPException(status_code=401, detail="Token inválido")

# --- API ENDPOINTS ---

@app.get("/api/auth/me")
async def auth_me(user=Depends(get_current_user)):
    """Permite al frontend conocer el rol del usuario sin hardcodear emails."""
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
        all_recs = [doc.to_dict() for doc in docs]
        
        if user["is_admin"]:
            return all_recs
        
        slug = user.get("inst_slug", "").lower()
        return [r for r in all_recs if slug in r.get("institution", "").lower()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(get_current_user)):
    """Diagnóstico del Error 500: Verificamos disponibilidad del modelo y cuotas."""
    try:
        # Validamos que el modelo exista en la configuración
        if not MODEL_NAME:
            raise ValueError("La variable GEMINI_MODEL_NAME no está configurada")
            
        model = GenerativeModel(MODEL_NAME)
        prompt = f"Analiza esta barrera institucional para el MNPT: {data.text}. Propón 3 soluciones."
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        logger.error(f"FALLO CRÍTICO IA ({MODEL_NAME}): {str(e)}")
        # Si el modelo gemini-2.5-flash no está disponible aún, devolvemos un error descriptivo
        raise HTTPException(
            status_code=500, 
            detail=f"Error en Vertex AI con el modelo {MODEL_NAME}. Verifique disponibilidad en {LOCATION}."
        )

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
        
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "id": sub_ref.id,
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "description": description,
            "file_url": blob.public_url,
            "status": "PENDIENTE",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Evidencia subida correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/pending")
async def list_pending(user=Depends(get_current_user)):
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Solo administradores")
    subs = db.collection("artifacts").document(APP_ID).collection("submissions").where("status", "==", "PENDIENTE").stream()
    return [s.to_dict() for s in subs]

@app.get("/api/report/generate")
async def generate_pdf(user=Depends(get_current_user)):
    recs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 750, "SIIS-SPT: Informe de Cumplimiento")
    y = 700
    for doc in recs:
        d = doc.to_dict()
        p.setFont("Helvetica-Bold", 10)
        p.drawString(50, y, f"[{d.get('id')}] {d.get('institution')}")
        p.setFont("Helvetica", 10)
        p.drawString(50, y-15, f"Avance: {d.get('progress', 0)}%")
        y -= 45
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
