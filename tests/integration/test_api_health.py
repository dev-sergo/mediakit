"""HTTP tests for GET /healthz.

httpx calls and Redis are mocked — no live dependencies needed.
Uses settings.comfyui_url so the mock works regardless of .env configuration.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import respx
from fastapi.testclient import TestClient

from mediakit.config import settings
from mediakit.server.app import app

_COMFYUI_STATS = f"{settings.comfyui_url}/system_stats"


def _redis_ok() -> AsyncMock:
    return AsyncMock(ping=AsyncMock(), aclose=AsyncMock())


def _redis_down() -> AsyncMock:
    m = AsyncMock()
    m.ping.side_effect = ConnectionError("refused")
    return m


def _health_client() -> TestClient:
    return TestClient(app)


def test_healthz_all_ok() -> None:
    with respx.mock:
        respx.get(_COMFYUI_STATS).mock(return_value=httpx.Response(200))
        with patch("mediakit.server.routes.health.aioredis.from_url", return_value=_redis_ok()):
            resp = _health_client().get("/healthz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["comfyui"] == "ok"
    assert body["checks"]["redis"] == "ok"


def test_healthz_comfyui_unreachable() -> None:
    with respx.mock:
        respx.get(_COMFYUI_STATS).mock(side_effect=httpx.ConnectError("unreachable"))
        with patch("mediakit.server.routes.health.aioredis.from_url", return_value=_redis_ok()):
            resp = _health_client().get("/healthz")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["comfyui"] == "unreachable"
    assert body["checks"]["redis"] == "ok"


def test_healthz_comfyui_non_200() -> None:
    with respx.mock:
        respx.get(_COMFYUI_STATS).mock(return_value=httpx.Response(503))
        with patch("mediakit.server.routes.health.aioredis.from_url", return_value=_redis_ok()):
            resp = _health_client().get("/healthz")

    assert resp.json()["checks"]["comfyui"] == "http_503"
    assert resp.json()["status"] == "degraded"


def test_healthz_redis_unreachable() -> None:
    with respx.mock:
        respx.get(_COMFYUI_STATS).mock(return_value=httpx.Response(200))
        with patch("mediakit.server.routes.health.aioredis.from_url", return_value=_redis_down()):
            resp = _health_client().get("/healthz")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"] == "unreachable"
    assert body["checks"]["comfyui"] == "ok"


def test_healthz_both_down() -> None:
    with respx.mock:
        respx.get(_COMFYUI_STATS).mock(side_effect=httpx.ConnectError("timeout"))
        with patch("mediakit.server.routes.health.aioredis.from_url", return_value=_redis_down()):
            resp = _health_client().get("/healthz")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["comfyui"] == "unreachable"
    assert body["checks"]["redis"] == "unreachable"
