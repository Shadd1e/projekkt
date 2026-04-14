"""
Shaddies Checker — Python Microservice
=======================================
FastAPI service that:
1. Receives a .docx file
2. Scores each paragraph for AI-likeness (perplexity)
3. Checks each paragraph against web + academic sources (plagiarism)
4. Paraphrases flagged paragraphs with DeepSeek
5. Runs a humanization pass on paraphrased output
6. Returns a corrected .docx + a JSON report

Run locally:
    uvicorn main:app --reload --port 8000

Environment variables required (.env):
    DEEPSEEK_API_KEY=...
    BRAVE_API_KEY=...
    INTERNAL_API_SECRET=...   (shared secret between Next.js and this service)
"""

import os
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from processor import process_document
from cleanup import schedule_cleanup

load_dotenv()

INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "change-this-secret")
TMP_DIR = Path("/tmp/shaddies")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Lifespan: start background cleanup loop ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(schedule_cleanup(TMP_DIR))
    yield
    task.cancel()

app = FastAPI(title="Shaddies Checker", version="1.0.0", lifespan=lifespan)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main processing endpoint ──────────────────────────────────────────────────
@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_internal_secret: str = Header(...),
):
    # Auth — only Next.js backend can call this
    if x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Validate file type
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are accepted.")

    # Validate file size (10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit.")

    # Save upload to /tmp
    job_id = str(uuid.uuid4())
    input_path  = TMP_DIR / f"{job_id}_input.docx"
    output_path = TMP_DIR / f"{job_id}_output.docx"

    input_path.write_bytes(contents)

    # Process (this is the heavy work — runs synchronously in a thread pool)
    try:
        report = await asyncio.to_thread(
            process_document,
            str(input_path),
            str(output_path),
        )
    except Exception as e:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # Clean up input file immediately
    input_path.unlink(missing_ok=True)

    return JSONResponse({
        "job_id": job_id,
        "report": report,
    })


# ── Download endpoint ─────────────────────────────────────────────────────────
@app.get("/download/{job_id}")
def download(job_id: str, x_internal_secret: str = Header(...)):
    if x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Sanitise job_id — must be a valid UUID, no path traversal
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
