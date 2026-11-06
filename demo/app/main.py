import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import MAX_REFS, Engine, PageError

HERE = Path(__file__).parent
(HERE / "samples").mkdir(exist_ok=True)
MAX_BYTES = 20 * 1024 * 1024
app = FastAPI(title="Handwriting verification demo")
engine = Engine()
app.mount("/samples", StaticFiles(directory=HERE / "samples"), name="samples")


async def _read(f: UploadFile) -> bytes:
    data = await f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File larger than 20 MB.")
    return data


@app.get("/api/health")
def health():
    return {"ok": True, "calibrated": engine.calib is not None}


@app.get("/api/info")
def info():
    c = engine.calib or {}
    s = HERE / "samples" / "samples.json"
    return {"model": engine.model_name, "k": 64, "max_refs": MAX_REFS, "calibration": c,
            "samples": json.load(open(s)) if s.exists() else []}


@app.post("/api/verify")
async def verify(reference: list[UploadFile] = File(...), questioned: UploadFile = File(...)):
    refs = [await _read(f) for f in reference]
    q = await _read(questioned)
    try:
        return engine.verify(refs, q)
    except PageError as e:
        return JSONResponse({"error": str(e)}, status_code=422)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")
