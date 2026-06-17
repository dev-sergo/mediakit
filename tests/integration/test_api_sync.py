"""HTTP tests for sync ops (compress, resize, convert, lqip, variants).

No ComfyUI or Redis needed — ops run in-process via Pillow.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mediakit.config import settings
from mediakit.server.app import app


# ─── compress ────────────────────────────────────────────────────────────────

def test_compress_webp_returns_200(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/compress",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"format": "webp"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "output_path" in body
    assert body["output_bytes"] > 0
    assert body["savings_pct"] is not None


def test_compress_quality_int(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/compress",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"format": "jpeg", "quality": "60"},
    )
    assert resp.status_code == 200


def test_compress_quality_preset(client: TestClient, jpeg_bytes: bytes) -> None:
    for preset in ("low", "medium", "high", "max"):
        resp = client.post(
            "/v1/ops/compress",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"format": "webp", "quality": preset},
        )
        assert resp.status_code == 200, f"Failed for quality={preset}"


def test_compress_max_width(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/compress",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"format": "webp", "max_width": "200"},
    )
    assert resp.status_code == 200


# ─── resize ──────────────────────────────────────────────────────────────────

def test_resize_fit(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/resize",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"width": "400", "height": "400", "mode": "fit"},
    )
    assert resp.status_code == 200
    assert "output_path" in resp.json()


def test_resize_fill(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/resize",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"width": "200", "height": "200", "mode": "fill"},
    )
    assert resp.status_code == 200


def test_resize_pad(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/resize",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"width": "500", "height": "500", "mode": "pad"},
    )
    assert resp.status_code == 200


# ─── convert ─────────────────────────────────────────────────────────────────

def test_convert_png_to_webp(client: TestClient, png_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/convert",
        files={"file": ("photo.png", png_bytes, "image/png")},
        data={"format": "webp"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["output_bytes"] > 0


# ─── lqip ────────────────────────────────────────────────────────────────────

def test_lqip_returns_data_url(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/lqip",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data_url"].startswith("data:image/webp;base64,")
    assert body["width"] > 0
    assert body["height"] > 0


def test_lqip_custom_size(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/lqip",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"size": "32"},
    )
    assert resp.status_code == 200


# ─── variants ────────────────────────────────────────────────────────────────

def test_variants_returns_list(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/variants",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"sizes": "400", "formats": "webp"},
    )
    assert resp.status_code == 200
    variants = resp.json()["variants"]
    assert len(variants) >= 1
    assert all("path" in v and "width" in v for v in variants)


def test_variants_multi_format(client: TestClient, jpeg_bytes: bytes) -> None:
    resp = client.post(
        "/v1/ops/variants",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"sizes": "400", "formats": "webp,jpeg"},
    )
    assert resp.status_code == 200
    fmts = {v["format"] for v in resp.json()["variants"]}
    assert "webp" in fmts
    assert "jpeg" in fmts


# ─── upload size limit ───────────────────────────────────────────────────────

def test_upload_too_large_returns_413(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "storage_uploads", tmp_path / "uploads")
    monkeypatch.setattr(settings, "storage_outputs", tmp_path / "outputs")
    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "storage_max_upload_mb", 1)  # 1 MB limit
    oversized = b"\x00" * (1 * 1024 * 1024 + 1)  # 1 MB + 1 byte
    c = TestClient(app)
    resp = c.post(
        "/v1/ops/compress",
        files={"file": ("big.jpg", oversized, "image/jpeg")},
        data={"format": "webp"},
    )
    assert resp.status_code == 413


# ─── auth ────────────────────────────────────────────────────────────────────

def test_auth_rejected_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jpeg_bytes: bytes
) -> None:
    monkeypatch.setattr(settings, "api_token", "secret-key")
    monkeypatch.setattr(settings, "storage_uploads", tmp_path / "uploads")
    monkeypatch.setattr(settings, "storage_outputs", tmp_path / "outputs")
    c = TestClient(app)
    resp = c.post(
        "/v1/ops/compress",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"format": "webp"},
    )
    assert resp.status_code == 401


def test_auth_accepted_with_valid_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jpeg_bytes: bytes
) -> None:
    monkeypatch.setattr(settings, "api_token", "secret-key")
    monkeypatch.setattr(settings, "storage_uploads", tmp_path / "uploads")
    monkeypatch.setattr(settings, "storage_outputs", tmp_path / "outputs")
    c = TestClient(app, headers={"Authorization": "Bearer secret-key"})
    resp = c.post(
        "/v1/ops/compress",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        data={"format": "webp"},
    )
    assert resp.status_code == 200
