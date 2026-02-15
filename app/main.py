import os
import json
import secrets
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from app.config import ADMIN_PASSWORD, UPLOAD_DIR, APP_ENV
from app.services import document_processor, vector_store, rag_chain

app = FastAPI(title="Aviation Advisor", docs_url=None, redoc_url=None)

security = HTTPBasic()

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct = secrets.compare_digest(
        credentials.password.encode("utf8"),
        ADMIN_PASSWORD.encode("utf8"),
    )
    if not correct:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# --- Chat ---


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
async def chat(request: ChatRequest, _user: str = Depends(verify_credentials)):
    async def generate():
        for text in rag_chain.stream_response(request.message, request.history):
            yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- Admin: Document Management ---


@app.post("/api/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    _user: str = Depends(verify_credentials),
):
    results = []
    for file in files:
        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process and index
        chunks = document_processor.process_document(file_path)
        if chunks:
            count = vector_store.add_documents(chunks, file.filename)
            results.append({"filename": file.filename, "chunks": count, "status": "indexed"})
        else:
            results.append({"filename": file.filename, "chunks": 0, "status": "no_text_extracted"})

    return {"results": results, "total_chunks": vector_store.get_total_chunks()}


@app.get("/api/documents")
async def list_documents(_user: str = Depends(verify_credentials)):
    sources = vector_store.get_document_sources()
    return {"documents": sources, "total_chunks": vector_store.get_total_chunks()}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, _user: str = Depends(verify_credentials)):
    deleted = vector_store.delete_document(filename)

    # Also remove the uploaded file
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"deleted_chunks": deleted, "total_chunks": vector_store.get_total_chunks()}


# --- Static files and pages ---

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(_user: str = Depends(verify_credentials)):
    with open(os.path.join(static_dir, "index.html")) as f:
        return HTMLResponse(content=f.read())


@app.get("/admin", response_class=HTMLResponse)
async def admin(_user: str = Depends(verify_credentials)):
    with open(os.path.join(static_dir, "admin.html")) as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "healthy", "chunks": vector_store.get_total_chunks()}
