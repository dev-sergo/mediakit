from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import sentry_sdk
import uvicorn
from fastapi import FastAPI

from mediakit.config import settings
from mediakit.jobs.queue import close_pool
from mediakit.logging import configure_logging
from mediakit.server.routes import health, jobs, ops, pipelines

if settings.sentry_dsn and settings.sentry_dsn.startswith("https://"):
    sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield
    await close_pool()


app = FastAPI(title="mediakit", version="0.1.0", docs_url="/docs", lifespan=lifespan)

app.include_router(health.router)
app.include_router(ops.router)
app.include_router(jobs.router)
app.include_router(pipelines.router)


def main() -> None:
    configure_logging(settings.log_level, settings.log_format)
    uvicorn.run(
        "mediakit.server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
