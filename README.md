# Super-Image-Resolution

A learning project exploring image restoration with a convolutional autoencoder and skip connections. Originally built using face images from LFW.

> **Scope:** This model restores a downsampled image at **80×80 pixels**. It does **not** increase the output image dimensions or reconstruct guaranteed real-world detail. A true super-resolution model and interactive UI are planned future improvements.

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

## Next iterations

1. Benchmark restoration quality using PSNR/SSIM on a held-out test set and compare against bicubic interpolation.
2. Add a FastAPI inference endpoint and React/TypeScript upload, compare and download interface.
3. Explore true upscaling and a larger, more diverse dataset, then add screenshots and a deployed demo.

## Credits

This project began as a Coursera-inspired learning experiment, with guidance from an article. Original references should be added when available.
