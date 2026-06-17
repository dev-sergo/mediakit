from typing import Any

import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter

from mediakit.config import settings

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, Any]:
    checks: dict[str, str] = {}

    # ComfyUI
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{settings.comfyui_url}/system_stats")
            checks["comfyui"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception:
        checks["comfyui"] = "unreachable"

    # Redis
    try:
        r_client = aioredis.from_url(settings.redis_url, socket_timeout=2)  # type: ignore[no-untyped-call]
        await r_client.ping()
        await r_client.aclose()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unreachable"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
