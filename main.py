#!/usr/bin/env python3
"""Train the face-image autoencoder or enhance one image with saved weights.

This is restoration at 80x80, not arbitrary-resolution upscaling.
"""
import argparse
from pathlib import Path

import numpy as np

from resolution import build_model, load_image, pixelate_image


def train(args):
    from sklearn.model_selection import train_test_split
    from tensorflow import keras

    paths = sorted(Path(args.data_dir).rglob("*.jpg"))
    if len(paths) < 10:
        raise ValueError("Need at least 10 JPG images in the training directory")
    originals = np.stack([load_image(p) for p in paths])
    low_res = np.stack([pixelate_image(im) for im in originals])
    x_train, x_val, y_train, y_val = train_test_split(
        low_res, originals, test_size=0.2, random_state=args.seed
    )
    model = build_model()
    model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
        ],
    )
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    print(f"Saved model to {args.model}")


def enhance(args):
    from PIL import Image
    from tensorflow import keras

    if not Path(args.model).is_file():
        raise FileNotFoundError(f"Model not found: {args.model}. Run train first.")
    model = keras.models.load_model(args.model, compile=False)
    image = load_image(args.image)
    low_res = pixelate_image(image)
    prediction = model.predict(low_res[None, ...], verbose=0)[0]
    output = (np.clip(prediction, 0, 1) * 255).round().astype(np.uint8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(args.output)
    print(f"Saved enhanced image to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="Train and save a model")
    train_parser.add_argument("--data-dir", default="lfw")
    train_parser.add_argument("--model", default="models/super_resolution.keras")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--seed", type=int, default=32)
    train_parser.set_defaults(handler=train)
    enhance_parser = sub.add_parser("enhance", help="Enhance with a saved model")
    enhance_parser.add_argument("image")
    enhance_parser.add_argument("--model", default="models/super_resolution.keras")
    enhance_parser.add_argument("--output", default="enhanced.png")
    enhance_parser.set_defaults(handler=enhance)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
