"""Shared fixtures for integration tests (HTTP layer, no real Redis/ComfyUI)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from mediakit.config import settings
from mediakit.server.app import app


@pytest.fixture(scope="session")
def jpeg_bytes() -> bytes:
    img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    img = Image.new("RGBA", (400, 300), color=(255, 128, 0, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def tmp_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "storage_uploads", tmp_path / "uploads")
    monkeypatch.setattr(settings, "storage_outputs", tmp_path / "outputs")
    monkeypatch.setattr(settings, "api_token", "")  # disable auth


@pytest.fixture()
def client(tmp_storage: None) -> TestClient:
    return TestClient(app)
