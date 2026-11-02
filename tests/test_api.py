import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from api import app as api_module

client = TestClient(api_module.app)


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (80, 80), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_returns_readiness_flag():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["model_available"], bool)


def test_rejects_non_image():
    response = client.post("/api/enhance", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_rejects_corrupt_image():
    response = client.post("/api/enhance", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 422


def test_rejects_oversized_upload():
    response = client.post("/api/enhance", files={"file": ("big.png", b"x" * (api_module.MAX_BYTES + 1), "image/png")})
    assert response.status_code == 413


def test_missing_model_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "missing.keras")
    response = client.post("/api/enhance", files={"file": ("valid.png", image_bytes(), "image/png")})
    assert response.status_code == 503


def test_enhance_returns_png(monkeypatch, tmp_path):
    import numpy as np

    class FakeModel:
        def predict(self, batch, verbose=0):
            return batch

    checkpoint = tmp_path / "model.keras"
    checkpoint.touch()
    monkeypatch.setattr(api_module, "MODEL_PATH", checkpoint)
    monkeypatch.setattr(api_module, "get_model", lambda: FakeModel())
    response = client.post("/api/enhance", files={"file": ("valid.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as result:
        assert result.size == (80, 80)


def test_degrade_returns_model_input_png():
    response = client.post("/api/degrade", files={"file": ("valid.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (80, 80)
