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

# --- CONFIGURACIÓN DE ENTORNO ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = os.getenv("APP_ID", "siis-spt-cr")
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")
LOCATION = "us-central1"

app = FastAPI(title="SIIS-SPT Sistema Integral")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización de Firebase y Vertex AI
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

# --- SEGURIDAD Y FILTRADO POR INSTITUCIÓN ---
async def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token ausente")
    token = auth_header.split("Bearer ")[1]
    try:
        user = auth.verify_id_token(token)
        email = user.get("email", "")
        # Es Admin si el dominio es mnpt.go.cr
        user["is_admin"] = email.endswith("@mnpt.go.cr")
        # Mapeo simple de institución basado en el dominio del correo
        user["institution_domain"] = email.split("@")[1].split(".")[0]
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Sesión inválida: {str(e)}")

# --- ENDPOINTS DE API ---

@app.get("/api/recommendations")
async def list_recommendations(user=Depends(get_current_user)):
    """
    Lista las recomendaciones. 
    Si no es admin, solo ve las que corresponden a su institución.
    """
    try:
        ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
        docs = ref.stream()
        all_recs = [doc.to_dict() for doc in docs]
        
        if user["is_admin"]:
            return all_recs
        
        # Filtrado para evitar contaminación de información
        # El sistema busca coincidencia entre el dominio del correo y la institución
        filtered = [r for r in all_recs if user["institution_domain"].lower() in r.get("institution", "").lower()]
        return filtered
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/analyze")
async def analyze_barrier(data: BarrierInput, user=Depends(get_current_user)):
    """Laboratorio de IA: Genera estrategias de incidencia política."""
    try:
        model = GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Actúa como un experto del Subcomité para la Prevención de la Tortura (SPT). "
            f"Analiza este obstáculo institucional en Costa Rica: '{data.text}'. "
            f"Propón 3 estrategias técnicas de incidencia para superarlo."
        )
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
    """Carga evidencias a Cloud Storage y crea ticket de validación."""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        file_path = f"evidence/{recommendation_id}/{file.filename}"
        blob = bucket.blob(file_path)
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        # Generar ticket para el Panel del Administrador
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "id": sub_ref.id,
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "description": description,
            "file_url": blob.public_url,
            "status": "PENDING",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Evidencia subida. El MNPT revisará el avance."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/generate")
async def generate_pdf(user=Depends(get_current_user)):
    """Genera el Informe Oficial de Cumplimiento en PDF."""
    try:
        docs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Encabezado
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, 750, "SIIS-SPT: Informe de Monitoreo Costa Rica")
        p.setFont("Helvetica", 10)
        p.drawString(50, 735, f"Generado el 2026-02-03 para: {user['email']}")
        p.line(50, 725, 550, 725)
        
        y = 700
        for doc in docs:
            d = doc.to_dict()
            p.setFont("Helvetica-Bold", 12)
            p.drawString(50, y, f"[{d.get('id', 'N/A')}] {d.get('institution', 'N/A')}")
            y -= 15
            p.setFont("Helvetica", 10)
            p.drawString(60, y, f"Avance: {d.get('progress', 0)}% - Estado: {d.get('status', 'N/A')}")
            y -= 35
            if y < 100:
                p.showPage()
                y = 750
        
        p.save()
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf", headers={
            "Content-Disposition": "attachment;filename=Informe_SIIS_SPT_CRI.pdf"
        })
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
