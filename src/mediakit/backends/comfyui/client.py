from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
import structlog
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from mediakit.backends.comfyui.exceptions import (
    ComfyUIError,
    ComfyUIExecutionError,
    ComfyUITimeoutError,
)

log = structlog.get_logger(__name__)

WorkflowDict = dict[str, Any]


class ComfyUIClient:
    """Async HTTP + WebSocket client for ComfyUI.

    Submits a workflow via POST /prompt, waits on /ws for the matching
    prompt_id to complete, then downloads output images from /history.
    """

    def __init__(
        self,
        base_url: str,
        *,
        output_dir: Path,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._output_dir = output_dir
        self._timeout = timeout_seconds
        self._client_id = str(uuid.uuid4())
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
        )

    @property
    def client_id(self) -> str:
        return self._client_id

    async def ping(self, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
        try:
            r = await self._http.get("/system_stats", timeout=timeout_seconds)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"ComfyUI not reachable at {self._base_url}: {exc}") from exc
        body = r.json()
        if not isinstance(body, dict):
            raise ComfyUIError(f"Malformed /system_stats response: {body!r}")
        return body

    async def free_memory(self) -> None:
        try:
            r = await self._http.post(
                "/free", json={"unload_models": True, "free_memory": True}, timeout=30.0
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to free VRAM: {exc}") from exc
        log.info("comfyui.free_memory")

    async def soft_free_memory(self) -> None:
        """Clear activation tensors without unloading models (no cold-start penalty)."""
        try:
            r = await self._http.post(
                "/free", json={"unload_models": False, "free_memory": True}, timeout=10.0
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to soft-free VRAM: {exc}") from exc
        log.info("comfyui.soft_free_memory")

    async def available_vram_mb(self) -> int:
        """Return free VRAM in MB on the primary GPU. Returns 0 on error."""
        try:
            stats = await self.ping(timeout_seconds=3.0)
            devices = stats.get("devices", [])
            if devices:
                return int(devices[0].get("vram_free", 0)) // (1024 * 1024)
        except ComfyUIError:
            pass
        return 0

    async def ensure_vram(self, min_mb: int) -> None:
        """If free VRAM < min_mb, unload all models and free memory.

        Call before heavy generation to avoid mid-job OOM when residual
        models from a previous job are still occupying VRAM.

        Retries /free up to 3 times with backoff — network (e.g. mobile
        hotspot) may drop briefly between generations.
        """
        free = await self.available_vram_mb()
        if free >= min_mb:
            return
        log.warning("comfyui.low_vram", free_mb=free, min_mb=min_mb)
        for attempt in range(3):
            try:
                await self.free_memory()
                return
            except ComfyUIError as exc:
                if attempt < 2:
                    wait = 10 * (attempt + 1)
                    log.warning(
                        "comfyui.free_memory_retry",
                        attempt=attempt + 1, wait_s=wait, error=str(exc),
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

    async def upload_image(self, path: Path, *, overwrite: bool = True, subfolder: str = "") -> str:
        if not path.is_file():
            raise ComfyUIError(f"Cannot upload, file not found: {path}")
        with path.open("rb") as fp:
            try:
                r = await self._http.post(
                    "/upload/image",
                    files={"image": (path.name, fp, "application/octet-stream")},
                    data={
                        "overwrite": "true" if overwrite else "false",
                        "type": "input",
                        "subfolder": subfolder,
                    },
                )
            except httpx.HTTPError as exc:
                raise ComfyUIError(f"Failed to upload {path}: {exc}") from exc
        if r.status_code != 200:
            raise ComfyUIError(f"Upload rejected (HTTP {r.status_code}): {r.text}")
        body = r.json()
        name = body.get("name")
        if not isinstance(name, str):
            raise ComfyUIError(f"Malformed /upload/image response: {body!r}")
        log.info("comfyui.image_uploaded", path=str(path), server_name=name)
        return name

    async def submit_workflow(self, workflow: WorkflowDict) -> str:
        payload = {"prompt": workflow, "client_id": self._client_id}
        try:
            r = await self._http.post("/prompt", json=payload)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to submit workflow: {exc}") from exc
        if r.status_code != 200:
            raise ComfyUIError(f"ComfyUI rejected workflow (HTTP {r.status_code}): {r.text}")
        data = r.json()
        prompt_id = data.get("prompt_id")
        if not isinstance(prompt_id, str):
            raise ComfyUIError(f"Malformed /prompt response: {data!r}")
        log.info("comfyui.workflow_submitted", prompt_id=prompt_id)
        return prompt_id

    async def wait_for_result(self, prompt_id: str) -> list[Path]:
        await self._await_completion(prompt_id)
        return await self._download_outputs(prompt_id)

    async def _await_completion(self, prompt_id: str) -> None:
        ws_url = self._build_ws_url()
        try:
            async with ws_connect(ws_url, max_size=2**24, open_timeout=10.0) as ws:
                if await self._already_complete(prompt_id):
                    return
                try:
                    async with asyncio.timeout(self._timeout):
                        async for raw in ws:
                            if isinstance(raw, bytes):
                                continue
                            event = json.loads(raw)
                            if self._is_error_event(event, prompt_id):
                                raise ComfyUIExecutionError(
                                    prompt_id=prompt_id, payload=event.get("data", {})
                                )
                            if self._is_terminal_event(event, prompt_id):
                                return
                except TimeoutError as exc:
                    raise ComfyUITimeoutError(
                        f"Workflow {prompt_id} did not complete within {self._timeout}s"
                    ) from exc
        except ConnectionClosed as exc:
            raise ComfyUIError(f"WebSocket closed before {prompt_id} finished: {exc}") from exc

    async def _already_complete(self, prompt_id: str) -> bool:
        try:
            r = await self._http.get(f"/history/{prompt_id}")
            r.raise_for_status()
        except httpx.HTTPError:
            return False
        return bool(r.json().get(prompt_id, {}).get("outputs"))

    @staticmethod
    def _is_terminal_event(event: dict[str, Any], prompt_id: str) -> bool:
        if event.get("type") != "executing":
            return False
        data = event.get("data", {})
        return data.get("prompt_id") == prompt_id and data.get("node") is None

    @staticmethod
    def _is_error_event(event: dict[str, Any], prompt_id: str) -> bool:
        if event.get("type") != "execution_error":
            return False
        return bool(event.get("data", {}).get("prompt_id") == prompt_id)

    async def _download_outputs(self, prompt_id: str) -> list[Path]:
        try:
            r = await self._http.get(f"/history/{prompt_id}")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to fetch history for {prompt_id}: {exc}") from exc

        history = r.json().get(prompt_id, {})
        outputs: dict[str, Any] = history.get("outputs", {})
        if not outputs:
            raise ComfyUIError(f"No outputs in history for prompt {prompt_id}")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for node_id, node_output in outputs.items():
            for image in node_output.get("images", []):
                saved.append(await self._download_image(prompt_id, node_id, image))

        if not saved:
            raise ComfyUIError(f"Prompt {prompt_id} produced no images")
        log.info("comfyui.outputs_downloaded", prompt_id=prompt_id, count=len(saved))
        return saved

    async def _download_image(self, prompt_id: str, node_id: str, image: dict[str, Any]) -> Path:
        try:
            r = await self._http.get(
                "/view",
                params={
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                },
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Failed to download {image['filename']}: {exc}") from exc
        target = self._output_dir / f"{prompt_id}_{node_id}_{image['filename']}"
        target.write_bytes(r.content)
        return target

    def _build_ws_url(self) -> str:
        if self._base_url.startswith("https://"):
            return f"wss://{self._base_url[8:]}/ws?clientId={self._client_id}"
        if self._base_url.startswith("http://"):
            return f"ws://{self._base_url[7:]}/ws?clientId={self._client_id}"
        raise ComfyUIError(f"Unsupported scheme: {self._base_url}")

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
