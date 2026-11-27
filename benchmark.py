"""Evaluate restoration against a bicubic interpolation baseline on held-out images."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from resolution import load_image, pixelate_image


def psnr(reference, prediction):
    """Peak signal-to-noise ratio in dB for normalized RGB images."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape:
        raise ValueError("Image shapes must match")
    mse = np.mean((reference - prediction) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(reference, prediction):
    """Structural similarity, using skimage's per-channel SSIM."""
    from skimage.metrics import structural_similarity

    if reference.shape != prediction.shape:
        raise ValueError("Image shapes must match")
    return float(structural_similarity(
        reference, prediction, channel_axis=-1, data_range=1.0
    ))


def bicubic_baseline(reference, scale_percent=40):
    height, width = reference.shape[:2]
    reduced_size = (max(1, round(width * scale_percent / 100)),
                    max(1, round(height * scale_percent / 100)))
    reduced = cv2.resize(reference, reduced_size, interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def evaluate(paths, model, output_path):
    """Save per-image measurements; no claims of improvement without results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        original = load_image(path)
        degraded = pixelate_image(original)
        baseline = bicubic_baseline(original)
        predicted = np.clip(model.predict(degraded[None, ...], verbose=0)[0], 0, 1)
        rows.append({
            "image": str(path),
            "baseline_psnr": psnr(original, baseline),
            "model_psnr": psnr(original, predicted),
            "baseline_ssim": ssim(original, baseline),
            "model_ssim": ssim(original, predicted),
        })
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "image", "baseline_psnr", "model_psnr", "baseline_ssim", "model_ssim"
        ])
        writer.writeheader()
        writer.writerows(rows)
    for metric in ("psnr", "ssim"):
        baseline_mean = np.mean([row[f"baseline_{metric}"] for row in rows])
        model_mean = np.mean([row[f"model_{metric}"] for row in rows])
        print(f"{metric.upper()}: bicubic={baseline_mean:.4f}, model={model_mean:.4f}")
    print(f"Saved per-image results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="lfw")
    parser.add_argument("--model", default="models/super_resolution.keras")
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--seed", type=int, default=32)
    args = parser.parse_args()
    from sklearn.model_selection import train_test_split
    from tensorflow import keras

    paths = sorted(Path(args.data_dir).rglob("*.jpg"))
    if len(paths) < 10:
        parser.error("Need at least 10 JPG images")
    # This split must match the training split; do not train on the test subset.
    _, test_paths = train_test_split(paths, test_size=0.2, random_state=args.seed)
    model = keras.models.load_model(args.model, compile=False)
    evaluate(test_paths, model, args.output)


if __name__ == "__main__":
    main()
