"""arq job queue — thin wrappers over arq's enqueue_job / job result polling."""
from __future__ import annotations

from typing import Any

import arq
import arq.connections
import arq.jobs

from mediakit.config import settings

_pool: arq.connections.ArqRedis | None = None


async def _get_pool() -> arq.connections.ArqRedis:
    global _pool
    if _pool is None:
        _pool = await arq.create_pool(arq.connections.RedisSettings.from_dsn(settings.redis_url))
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


async def enqueue(
    function: str,
    *args: Any,
    job_id: str | None = None,
    **kwargs: Any,
) -> str:
    pool = await _get_pool()
    job = await pool.enqueue_job(function, *args, _job_id=job_id, **kwargs)
    if job is None:
        raise RuntimeError(f"Failed to enqueue {function} (duplicate job_id?)")
    return job.job_id


async def get_job_result(job_id: str) -> dict[str, Any] | None:
    pool = await _get_pool()
    job = arq.jobs.Job(job_id, redis=pool)
    job_status = await job.status()
    if job_status == arq.jobs.JobStatus.not_found:
        return None
    info = await job.info()
    return {
        "job_id": job_id,
        "status": job_status.value,
        "result": info.result if (info and hasattr(info, "result")) else None,
        "enqueue_time": info.enqueue_time.isoformat() if (info and info.enqueue_time) else None,
    }
