# Backend setup

From the repository root, use Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -r api/requirements.txt
```

Prepare your own LFW face-image JPGs in `lfw/` and train the residual model:

```bash
python main.py train --data-dir lfw --architecture residual --epochs 16 --model models/super_resolution.keras
```

Start the API:

```bash
uvicorn api.app:app --reload
```

The React app runs separately from `frontend/` with `npm install && npm run dev`. Vite proxies `/api` to port 8000. If using a different checkpoint, set the `MODEL_PATH` environment variable before starting the API.

**Endpoints:** `GET /api/health` reports model availability. `POST /api/degrade` returns the exact degraded model input, and `POST /api/enhance` returns the restored output. Both accept JPEG, PNG, or WebP uploads up to 10 MB and return 80 × 80 PNGs.

This is an experimental face-image restoration demo. It resizes uploads to 80 × 80, artificially degrades them, and applies the trained model. It is not a general-purpose upscaler.
