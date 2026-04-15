"""
Shaddies Checker — Python Microservice
=======================================
FastAPI service that:
1. /analyse  — quick scan: word count, tables, images, price quote
2. /process  — full pipeline: AI detection, plagiarism, paraphrase, humanize
3. /download — serve processed file

Environment variables required (.env):
    ANTHROPIC_API_KEY=...
    DEEPSEEK_API_KEY=...
    BRAVE_API_KEY=...
    INTERNAL_API_SECRET=...
"""

import os
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from processor import process_document, analyse_document
from cleanup import schedule_cleanup

load_dotenv()

INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "change-this-secret")
TMP_DIR = Path("/tmp/shaddies")
TMP_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(schedule_cleanup(TMP_DIR))
    yield
    task.cancel()

app = FastAPI(title="Shaddies Checker", version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Quick analysis endpoint (no AI calls, instant) ────────────────────────────
@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    x_internal_secret: str = Header(...),
):
    if x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are accepted.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit.")

    job_id    = str(uuid.uuid4())
    tmp_path  = TMP_DIR / f"{job_id}_scan.docx"
    tmp_path.write_bytes(contents)

    try:
        result = await asyncio.to_thread(analyse_document, str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(result)


# ── Full processing endpoint ──────────────────────────────────────────────────
@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_internal_secret: str = Header(...),
):
    if x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are accepted.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit.")

    job_id      = str(uuid.uuid4())
    input_path  = TMP_DIR / f"{job_id}_input.docx"
    output_path = TMP_DIR / f"{job_id}_output.docx"
    input_path.write_bytes(contents)

    try:
        report = await asyncio.to_thread(process_document, str(input_path), str(output_path))
    except Exception as e:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    input_path.unlink(missing_ok=True)
    return JSONResponse({"job_id": job_id, "report": report})


# ── Download endpoint ─────────────────────────────────────────────────────────
@app.get("/download/{job_id}")
def download(job_id: str, x_internal_secret: str = Header(...)):
    if x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID.")

    output_path = TMP_DIR / f"{job_id}_output.docx"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File not found or already expired.")

    return FileResponse(
        path=str(output_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="corrected_document.docx",
    )
