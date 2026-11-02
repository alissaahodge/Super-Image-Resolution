"""FastAPI inference service. Start with: uvicorn api.app:app --reload."""
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from resolution import IMAGE_SIZE, pixelate_image

MAX_BYTES = 10 * 1024 * 1024
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/super_resolution.keras"))
app = FastAPI(title="Image Restoration Studio", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_model():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH, compile=False)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.is_file()}


@app.post("/api/enhance")
async def enhance(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB upload limit.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.verify()
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB").resize(IMAGE_SIZE)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(422, "The file is not a valid image.")
    if not MODEL_PATH.is_file():
        raise HTTPException(503, "No trained model found. Train a model and set MODEL_PATH.")
    try:
        degraded = pixelate_image(rgb)
        prediction = get_model().predict(degraded[None, ...], verbose=0)[0]
        result = np.clip(prediction, 0, 1)
        output = Image.fromarray(np.rint(result * 255).astype(np.uint8), "RGB")
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
    except Exception as exc:
        raise HTTPException(500, "Enhancement failed. Check server logs.") from exc
    return Response(buffer.getvalue(), media_type="image/png", headers={
        "Content-Disposition": 'attachment; filename="enhanced.png"',
        "Cache-Control": "no-store",
    })
