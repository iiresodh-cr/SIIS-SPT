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

app = FastAPI(title="SIIS-SPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location="us-central1")

# --- SEGURIDAD ---
async def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401)
    token = auth_header.split("Bearer ")[1]
    try:
        user = auth.verify_id_token(token)
        # Es administrador si el correo termina en @mnpt.go.cr
        user["is_admin"] = user.get("email", "").endswith("@mnpt.go.cr")
        # Extraemos el dominio para el filtrado institucional
        user["institution_domain"] = user.get("email", "").split("@")[1].split(".")[0]
        return user
    except:
        raise HTTPException(status_code=401)

# --- ENDPOINTS ---

@app.get("/api/recommendations")
async def list_recommendations(user=Depends(get_current_user)):
    """Evita la contaminación: Filtra por institución si no es admin."""
    try:
        ref = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations")
        docs = ref.stream()
        all_recs = [doc.to_dict() for doc in docs]
        
        if user["is_admin"]:
            return all_recs
        
        # Filtro: El dominio del correo debe estar contenido en el nombre de la institución
        domain = user["institution_domain"].lower()
        return [r for r in all_recs if domain in r.get("institution", "").lower()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/analyze")
async def analyze_with_gemini(request: Request, user=Depends(get_current_user)):
    try:
        data = await request.json()
        model = GenerativeModel("gemini-1.5-flash")
        prompt = f"Como experto en el OPCAT para Costa Rica, analiza esta barrera: {data['text']}. Da 3 estrategias de incidencia."
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
        blob = bucket.blob(f"evidence/{recommendation_id}/{file.filename}")
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        # Registro para revisión del MNPT
        sub_ref = db.collection("artifacts").document(APP_ID).collection("submissions").document()
        sub_ref.set({
            "recommendation_id": recommendation_id,
            "submitted_by": user["email"],
            "file_url": blob.public_url,
            "description": description,
            "status": "PENDING",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "Evidencia subida para validación del MNPT"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/generate")
async def generate_pdf(user=Depends(get_current_user)):
    try:
        recs = db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("recommendations").stream()
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 750, "SIIS-SPT: Informe de Monitoreo")
        
        y = 710
        for doc in recs:
            d = doc.to_dict()
            p.setFont("Helvetica-Bold", 10)
            p.drawString(100, y, f"[{d.get('id')}] {d.get('institution')}")
            p.setFont("Helvetica", 10)
            p.drawString(100, y-15, f"Avance: {d.get('progress', 0)}% - Estado: {d.get('status', 'Pendiente')}")
            y -= 45
            if y < 100:
                p.showPage()
                y = 750
        p.save()
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")
