import os
import io
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

# --- CONFIGURACIÓN ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "siis-stp")
APP_ID = os.getenv("APP_ID", "siis-spt-cr")
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", f"{PROJECT_ID}.firebasestorage.app")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location="us-central1")

# --- SEGURIDAD ---
async def get_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401)
    token = auth_header.split("Bearer ")[1]
    try:
        # Aquí verificamos el token y podemos extraer claims de rol/institución
        return auth.verify_id_token(token)
    except:
        raise HTTPException(status_code=401)

# --- API ---
@app.get("/api/recommendations")
async def list_recommendations(user=Depends(get_user)):
    """Si es admin ve todo. Si es institución, solo ve lo suyo."""
    try:
        query = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
        
        # Lógica de prevención de contaminación
        if user.get("role") != "admin":
            # Filtra por la institución asignada al usuario
            docs = query.where("institution", "==", user.get("institution")).stream()
        else:
            docs = query.stream()
            
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/generate")
async def make_pdf(user=Depends(get_user)):
    """Genera el informe oficial para el SPT."""
    docs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 750, "INFORME SIIS-SPT: CUMPLIMIENTO COSTA RICA")
    y = 720
    for doc in docs:
        d = doc.to_dict()
        p.setFont("Helvetica-Bold", 10)
        p.drawString(100, y, f"[{d['id']}] {d['institution']}")
        p.setFont("Helvetica", 10)
        p.drawString(100, y-15, f"Estado: {d['status']} - Avance: {d['progress']}%")
        y -= 40
    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf")

# Servir Frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
