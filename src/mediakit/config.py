from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ComfyUI backend
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_timeout_s: int = 600  # 10 min — matches image-service default; Qwen takes ~4 min

    # Redis / arq
    redis_url: str = "redis://localhost:6379/0"

    # Local storage
    storage_uploads: Path = Path("./storage/uploads")
    storage_outputs: Path = Path("./storage/outputs")
    storage_max_upload_mb: int = 20

    # HTTP server
    api_token: str = Field(default="", description="Bearer token for HTTP auth")
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000

    # Worker
    worker_concurrency_gpu: int = 1  # RTX 3090 — one inference at a time

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"

    # Sentry
    sentry_dsn: str = ""

    # Video generation timeout — much longer than image ops (5–15 min per job)
    video_timeout_s: int = 900

    # ComfyUI model directory — used by models_registry to verify models on startup.
    # Leave unset to skip the check (default behaviour).
    comfyui_models_dir: Path | None = None

    def ensure_storage_dirs(self) -> None:
        self.storage_uploads.mkdir(parents=True, exist_ok=True)
        self.storage_outputs.mkdir(parents=True, exist_ok=True)


settings = Settings()
