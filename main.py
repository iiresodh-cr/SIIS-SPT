import os
import io
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
        decoded = auth.verify_id_token(token)
        # Identificamos si es administrador por el dominio del correo
        decoded["is_admin"] = decoded.get("email", "").endswith("@mnpt.go.cr")
        return decoded
    except:
        raise HTTPException(status_code=401)

# --- ENDPOINTS ---

@app.get("/api/recommendations")
async def list_recommendations(user=Depends(get_user)):
    """Evita la contaminación: Usuarios normales solo ven lo de su institución."""
    ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
    docs = ref.stream()
    all_recs = [doc.to_dict() for doc in docs]
    
    if user["is_admin"]:
        return all_recs
    
    # Lógica de filtrado institucional para usuarios no-admin
    # Se asume que el correo o un atributo del usuario indica su institución
    user_inst = user.get("email", "").split("@")[1].split(".")[0] # Ejemplo simple
    return [r for r in all_recs if user_inst in r.get("institution", "").lower()]

@app.post("/api/ai/analyze")
async def analyze(data: dict, user=Depends(get_user)):
    model = GenerativeModel("gemini-1.5-flash")
    prompt = f"Como asesor del MNPT Costa Rica, analiza esta barrera: {data['text']}. Da 3 soluciones."
    return {"analysis": model.generate_content(prompt).text}

@app.post("/api/evidence/upload")
async def upload(
    recommendation_id: str = Form(...), 
    description: str = Form(...),
    file: UploadFile = File(...), 
    user=Depends(get_user)
):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"evidence/{recommendation_id}/{file.filename}")
    blob.upload_from_file(file.file, content_type=file.content_type)
    
    # Registro para revisión del administrador
    sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
    sub_ref.set({
        "recommendation_id": recommendation_id,
        "submitted_by": user["email"],
        "file_url": blob.public_url,
        "description": description,
        "status": "PENDING",
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    return {"message": "Evidencia subida. Pendiente de validación MNPT."}

@app.get("/api/report/generate")
async def report(user=Depends(get_user)):
    """Genera el informe oficial en PDF."""
    recs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Informe de Cumplimiento SIIS-SPT")
    y = 720
    for doc in recs:
        d = doc.to_dict()
        p.setFont("Helvetica-Bold", 10)
        p.drawString(100, y, f"[{d['id']}] {d['institution']}")
        p.setFont("Helvetica", 10)
        p.drawString(100, y-15, f"Progreso: {d.get('progress', 0)}% - Estado: {d.get('status', 'Pendiente')}")
        y -= 45
        if y < 100: p.showPage(); y = 750
    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("static/index.html")
