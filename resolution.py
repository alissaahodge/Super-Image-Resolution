"""Image preprocessing and model construction for the original 80x80 autoencoder."""
from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = (80, 80)


def load_image(path):
    """Load an RGB image as float32 pixels in [0, 1]."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        return np.asarray(image, dtype=np.float32) / 255.0


def pixelate_image(image, scale_percent=40):
    """Downsample then restore the *exact* original spatial dimensions."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an H x W x 3 RGB image")
    if not 0 < scale_percent <= 100:
        raise ValueError("scale_percent must be in (0, 100]")
    height, width = image.shape[:2]
    reduced = (max(1, round(width * scale_percent / 100)),
               max(1, round(height * scale_percent / 100)))
    small = cv2.resize(image, reduced, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def build_residual_model():
    """Predict a correction to the degraded input, starting from identity."""
    from tensorflow import keras

    layers = keras.layers
    inputs = layers.Input(shape=(80, 80, 3))
    x = layers.Conv2D(48, 3, padding="same", activation="relu")(inputs)
    for _ in range(4):
        shortcut = x
        x = layers.Conv2D(48, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(48, 3, padding="same")(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation("relu")(x)
    correction = layers.Conv2D(
        3, 3, padding="same", kernel_initializer="zeros",
        bias_initializer="zeros"
    )(x)
    outputs = layers.Add()([inputs, correction])
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss="mse")
    return model


def build_model():
    """Build the original skip-connected convolutional autoencoder."""
    from tensorflow import keras

    layers = keras.layers
    inputs = layers.Input(shape=(80, 80, 3))
    x1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    x2 = layers.Conv2D(64, 3, activation="relu", padding="same")(x1)
    x3 = layers.MaxPool2D(padding="same")(x2)
    x4 = layers.Conv2D(128, 3, activation="relu", padding="same")(x3)
    x5 = layers.Conv2D(128, 3, activation="relu", padding="same")(x4)
    x6 = layers.MaxPool2D(padding="same")(x5)
    encoded = layers.Conv2D(256, 3, activation="relu", padding="same")(x6)
    x7 = layers.UpSampling2D()(encoded)
    x8 = layers.Conv2D(128, 3, activation="relu", padding="same")(x7)
    x9 = layers.Conv2D(128, 3, activation="relu", padding="same")(x8)
    x10 = layers.Add()([x5, x9])
    x11 = layers.UpSampling2D()(x10)
    x12 = layers.Conv2D(64, 3, activation="relu", padding="same")(x11)
    x13 = layers.Conv2D(64, 3, activation="relu", padding="same")(x12)
    x14 = layers.Add()([x2, x13])
    outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x14)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
