# Restore Studio

An interactive **AI image-restoration demo** built with React, TypeScript, FastAPI, and TensorFlow. Upload a photo, enhance it with a residual CNN, compare the before/after images, and download the result.

> **Scope:** Experimental restoration of 80 × 80 face images, not full-resolution photo upscaling. The model cannot recover guaranteed real-world details.

## Preview

Screenshots of the running React app will be added before PR #4 is marked ready. For now, see the [original project's screenshot](Screenshot%202021-12-31%20at%2012.08.00%20PM.png).

## How it works

1. **Upload** a JPG, PNG, or WebP image (up to 10 MB).
2. **Enhance** it through the FastAPI inference endpoint and TensorFlow model.
3. **Compare** the result with the interactive slider, then download a PNG.

The model uses a residual CNN to predict image corrections. On 120 held-out LFW images, it achieved **36.54 dB PSNR and 0.9740 SSIM**, compared with **34.31 dB and 0.9599** for bicubic interpolation. Results are reproducible through the [real-image evaluation workflow](.github/workflows/real-image-evaluation.yml).

## Run locally

Requires Python 3.11 and Node.js 20+. Trained model weights are **not** included in the repository.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -r api/requirements.txt

# Train with your own LFW JPG images in lfw/
python main.py train --data-dir lfw --architecture residual --epochs 16
uvicorn api.app:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. If you already have trained weights, set `MODEL_PATH` to the checkpoint path instead of retraining.

## Project structure

- `frontend/`: React + TypeScript upload, comparison, and download UI
- `api/`: FastAPI image validation and TensorFlow inference
- `resolution.py`, `main.py`: model architecture, preprocessing, and training
- `benchmark.py`: PSNR/SSIM evaluation against bicubic
- `tests/`, `.github/workflows/`: automated checks and reproducible evaluation

## Background

This project began as a face-restoration learning experiment using LFW images. It has since been modernized with a residual model, reproducible benchmarks, and an interactive web interface.
