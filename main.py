"""
Shaddies Checker — Python Microservice
=======================================
FastAPI service that:
1. /analyse       — quick scan: word count, tables, images, price quote
2. /prescan       — starts a background detection job; returns job_id immediately
3. /job-status    — poll for prescan result
4. /process       — full pipeline: AI detection, plagiarism, paraphrase, humanize
5. /download      — serve processed file

Environment variables required (.env):
    HF_API_KEY=...
    DEEPSEEK_API_KEY=...
    BRAVE_API_KEY=...
    INTERNAL_API_SECRET=...
"""

import os
import uuid
import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from processor import (
    process_document,
    analyse_document,
    prescan_document,
    prescan_document_async,
    extract_paragraphs,
    MAX_PARAGRAPHS,
)
from cleanup import schedule_cleanup

load_dotenv()

INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "change-this-secret")
TMP_DIR = Path("/tmp/shaddies")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────────────
# Railway is a persistent server — this dict survives across requests.
# Keys: railway job_id (str)
# Values: { "status": "processing"|"done"|"failed", "result": dict|None,
#            "error": str|None, "created_at": float }
_prescan_jobs: dict = {}


def _cleanup_old_jobs():
    """Remove jobs older than 2 hours to prevent unbounded memory growth."""
    cutoff = time.time() - 7200
    stale  = [jid for jid, j in _prescan_jobs.items() if j["created_at"] < cutoff]
    for jid in stale:
        _prescan_jobs.pop(jid, None)


async def _run_prescan_job(job_id: str, filepath: Path):
    """Background task: run the async prescan pipeline, update job store."""
    try:
        result = await prescan_document_async(str(filepath))
        _prescan_jobs[job_id] = {**_prescan_jobs[job_id], "status": "done", "result": result}
    except ValueError as e:
        # Paragraph limit exceeded or document unreadable
        _prescan_jobs[job_id] = {**_prescan_jobs[job_id], "status": "failed", "error": str(e)}
    except Exception as e:
        _prescan_jobs[job_id] = {**_prescan_jobs[job_id], "status": "failed", "error": "Scan failed internally."}
        print(f"[prescan job {job_id}] unexpected error: {e}")
    finally:
        filepath.unlink(missing_ok=True)
        _cleanup_old_jobs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if INTERNAL_API_SECRET == "change-this-secret":
        raise RuntimeError("INTERNAL_API_SECRET must be set to a secure value in environment variables.")
    task = asyncio.create_task(schedule_cleanup(TMP_DIR))
    yield
    task.cancel()


app = FastAPI(title="Shaddies Checker", version="3.0.0", lifespan=lifespan)


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

    job_id   = str(uuid.uuid4())
    tmp_path = TMP_DIR / f"{job_id}_scan.docx"
    tmp_path.write_bytes(contents)

    try:
        result = await asyncio.to_thread(analyse_document, str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(result)


# ── Pre-scan: start job, return immediately ───────────────────────────────────
@app.post("/prescan")
async def prescan(
    file: UploadFile = File(...),
    x_internal_secret: str = Header(...),
):
    """
    Accepts a .docx file, validates it, then immediately returns a job_id.
    The actual scan runs as a background asyncio task.
    Poll /job-status/{job_id} for results.
    """
    if x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are accepted.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit.")

    # Save to temp file so we can read it in the background task
    job_id   = str(uuid.uuid4())
    tmp_path = TMP_DIR / f"{job_id}_prescan.docx"
    tmp_path.write_bytes(contents)

    # Quick paragraph count check — reject before starting any AI work
    try:
        paragraphs = extract_paragraphs(str(tmp_path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not read document.")

    if len(paragraphs) > MAX_PARAGRAPHS:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Document has {len(paragraphs)} checkable paragraphs "
                f"(max {MAX_PARAGRAPHS} — please submit one chapter at a time)."
            ),
        )

    # Register job in store and fire off background task
    _prescan_jobs[job_id] = {
        "status":     "processing",
        "result":     None,
        "error":      None,
        "created_at": time.time(),
    }
    asyncio.create_task(_run_prescan_job(job_id, tmp_path))

    return JSONResponse({"job_id": job_id, "status": "processing"}, status_code=202)


# ── Pre-scan status poll ──────────────────────────────────────────────────────
@app.get("/job-status/{job_id}")
def job_status(job_id: str, x_internal_secret: str = Header(...)):
    if x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    job = _prescan_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] == "done":
        return JSONResponse({"status": "done", "result": job["result"]})
    if job["status"] == "failed":
        return JSONResponse({"status": "failed", "error": job.get("error", "Scan failed.")})

    return JSONResponse({"status": "processing"})


# ── Full processing endpoint ──────────────────────────────────────────────────
@app.post("/process")
async def process(
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
        report = await asyncio.wait_for(
            asyncio.to_thread(process_document, str(input_path), str(output_path)),
            timeout=300,
        )
    except asyncio.TimeoutError:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=504, detail="Processing timed out. Try a shorter document.")
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
