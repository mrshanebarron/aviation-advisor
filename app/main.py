import os
import json
import secrets
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.config import ADMIN_PASSWORD, UPLOAD_DIR, APP_ENV
from app.services import document_processor, vector_store, rag_chain

app = FastAPI(title="Aviation Advisor", docs_url=None, redoc_url=None)

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session tokens (in-memory, reset on restart)
_sessions: set[str] = set()


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    _sessions.add(token)
    return token


def verify_session(request: Request):
    token = request.cookies.get("session")
    if not token or token not in _sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Auth ---


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # If already logged in, redirect to home
    token = request.cookies.get("session")
    if token and token in _sessions:
        return RedirectResponse("/", status_code=302)

    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    with open(os.path.join(static_dir, "login.html")) as f:
        return HTMLResponse(content=f.read())


@app.post("/login")
async def login(password: str = Form(...)):
    correct = secrets.compare_digest(
        password.encode("utf8"),
        ADMIN_PASSWORD.encode("utf8"),
    )
    if not correct:
        return HTMLResponse(
            content='<script>window.location="/login?error=1"</script>',
            status_code=401,
        )

    token = create_session()
    response = RedirectResponse("/", status_code=302)
    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=86400 * 7)
    return response


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        _sessions.discard(token)
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session")
    return response


# --- Chat ---


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
async def chat(request: ChatRequest, req: Request):
    verify_session(req)

    async def generate():
        for text in rag_chain.stream_response(request.message, request.history):
            yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- Admin: Document Management ---


@app.post("/api/documents/upload")
async def upload_documents(req: Request, files: list[UploadFile] = File(...)):
    verify_session(req)

    results = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        chunks = document_processor.process_document(file_path)
        if chunks:
            count = vector_store.add_documents(chunks, file.filename)
            results.append({"filename": file.filename, "chunks": count, "status": "indexed"})
        else:
            results.append({"filename": file.filename, "chunks": 0, "status": "no_text_extracted"})

    return {"results": results, "total_chunks": vector_store.get_total_chunks()}


@app.get("/api/documents")
async def list_documents(req: Request):
    verify_session(req)
    sources = vector_store.get_document_sources()
    return {"documents": sources, "total_chunks": vector_store.get_total_chunks()}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, req: Request):
    verify_session(req)
    deleted = vector_store.delete_document(filename)

    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"deleted_chunks": deleted, "total_chunks": vector_store.get_total_chunks()}


# --- Static files and pages ---

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    token = request.cookies.get("session")
    if not token or token not in _sessions:
        return RedirectResponse("/login", status_code=302)
    with open(os.path.join(static_dir, "index.html")) as f:
        return HTMLResponse(content=f.read())


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    token = request.cookies.get("session")
    if not token or token not in _sessions:
        return RedirectResponse("/login", status_code=302)
    with open(os.path.join(static_dir, "admin.html")) as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "healthy", "chunks": vector_store.get_total_chunks()}
