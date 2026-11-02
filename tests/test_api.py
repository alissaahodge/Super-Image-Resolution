import pytest

pytest.importorskip('fastapi')
pytest.importorskip('multipart')
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_returns_readiness_flag():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["model_available"], bool)


def test_rejects_non_image():
    response = client.post("/api/enhance", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_rejects_invalid_image():
    response = client.post("/api/enhance", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 503
