# Restore Studio · Image Restoration

An interactive image-restoration project built with **React + TypeScript, FastAPI, and TensorFlow**. Upload a portrait, run a trained residual CNN, compare before/after with a draggable slider, and download the resulting PNG.

> **Important:** This is experimental **80×80 face-image restoration**, not arbitrary high-resolution upscaling. The model is not trained to reconstruct factual details from a low-resolution source.

## App preview

The application includes a responsive upload workspace, model-availability indicator, interactive before/after slider, and PNG export.

**Screenshots:** Actual running-app screenshots will be captured and committed before PR #4 is marked ready for review. The historic project screenshot below is not a screenshot of the new React UI.

## Run the web application

Use Python 3.11 and Node.js 20 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r api/requirements.txt
```

Train a model if you do not already have one saved locally:

```bash
python main.py train --data-dir lfw --architecture residual --epochs 16 --model models/super_resolution.keras
```

Start the API from the repository root:

```bash
uvicorn api.app:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite development server proxies `/api` requests to FastAPI. The API's `GET /api/health` reports whether the model file exists; `POST /api/enhance` accepts JPEG, PNG, or WebP and returns an 80×80 PNG. Uploaded images are processed in memory.

**Model weights are not committed.** A successful build does not mean inference is available until a trained `.keras` file is present. Set `MODEL_PATH` to use a different checkpoint.

## Benchmark

On the same 120 held-out LFW images, the residual model achieved **36.54 dB PSNR / 0.9740 SSIM**, compared with bicubic's **34.31 dB / 0.9599 SSIM**. See the reproducible GitHub Actions real-image quality evaluation for methodology and downloadable comparisons.

---

## Original project notes

A learning project exploring image restoration with a convolutional autoencoder and skip connections. Originally built using face images from LFW.

> The original project explored a convolutional autoencoder; the newer residual architecture is selectable with `--architecture residual`.

![Original project screenshot](Screenshot%202021-12-31%20at%2012.08.00%20PM.png)

## Setup

Use Python 3.11 and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Train once

The original LFW sample images should be placed under `lfw/`:

```bash
python main.py train --data-dir lfw --epochs 100
```

The trained model is saved to `models/super_resolution.keras` by default. Training can take time, depending on your hardware.

## Enhance an image (without retraining)

```bash
python main.py enhance path/to/photo.jpg --output enhanced.png
```

The input is resized to 80×80 and downsampled to simulate degradation before restoration. The output is 80×80. For now, inference is a demonstration of the learned restoration task, **not** a general photo enhancement tool.

## Tests

```bash
python -m pytest -q
```

Unit tests verify preprocessing shape preservation and input validation. GitHub Actions runs these tests without requiring a heavyweight model download. Training quality and TensorFlow runtime compatibility still require an end-to-end validation run.

## Architecture

- `resolution.py`: image loading, degradation and model architecture
- `main.py train`: dataset loading, train/validation split, training and model saving
- `main.py enhance`: saved-model inference and PNG export
- `tests/`: preprocessing unit tests

## Evaluate image quality

After training a fresh model with this version, run:

```bash
python benchmark.py --data-dir lfw --model models/super_resolution.keras
```

This compares model output against a bicubic interpolation baseline on a held-out 20% test split. The training pipeline uses a separate 60% training / 20% validation / 20% test split, based on sorted paths and the same seed. The benchmark writes per-image PSNR (dB) and SSIM values to `results/benchmark.csv`, and prints averages. Higher is better for both metrics. **Do not benchmark a model trained with the previous 80/20 split:** it may have trained on these test images.

Real-image benchmarks are now automated; see the results above.

## Next iterations

1. Benchmark restoration quality using PSNR/SSIM on a held-out test set and compare against bicubic interpolation.
2. Add a FastAPI inference endpoint and React/TypeScript upload, compare and download interface.
3. Explore true upscaling and a larger, more diverse dataset, then add screenshots and a deployed demo.

## Credits

This project began as a Coursera-inspired learning experiment, with guidance from an article. Original references should be added when available.
