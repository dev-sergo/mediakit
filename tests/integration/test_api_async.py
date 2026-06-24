"""HTTP tests for async AI ops (txt2img, img_edit, bg_remove, upscale) and pipelines.

enqueue and get_job_result are mocked — no Redis connection needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

# ─── AI ops — enqueue → 202 ──────────────────────────────────────────────────


def test_txt2img_enqueues_job(client: TestClient) -> None:
    with patch(
        "mediakit.server.routes.jobs.enqueue", new_callable=AsyncMock, return_value="job-txt2img"
    ):
        resp = client.post("/v1/ops/txt2img", data={"prompt": "a sunset over the ocean"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] == "job-txt2img"
    assert body["status"] == "queued"


def test_txt2img_random_seed_when_minus_one(client: TestClient) -> None:
    captured: list[tuple[object, ...]] = []

    async def capture_enqueue(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "job-seed"

    with patch("mediakit.server.routes.jobs.enqueue", side_effect=capture_enqueue):
        client.post("/v1/ops/txt2img", data={"prompt": "test", "seed": "-1"})

    assert captured
    _, params = captured[0]
    assert isinstance(params["seed"], int)
    assert params["seed"] != -1  # should have been replaced with a random value


def test_img_edit_enqueues_job(client: TestClient, jpeg_bytes: bytes) -> None:
    with patch(
        "mediakit.server.routes.jobs.enqueue", new_callable=AsyncMock, return_value="job-edit"
    ):
        resp = client.post(
            "/v1/ops/img-edit",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"prompt": "white studio background"},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-edit"


def test_bg_remove_enqueues_job(client: TestClient, jpeg_bytes: bytes) -> None:
    with patch(
        "mediakit.server.routes.jobs.enqueue", new_callable=AsyncMock, return_value="job-bg"
    ):
        resp = client.post(
            "/v1/ops/bg-remove",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-bg"


def test_upscale_enqueues_job(client: TestClient, jpeg_bytes: bytes) -> None:
    with patch(
        "mediakit.server.routes.jobs.enqueue", new_callable=AsyncMock, return_value="job-up"
    ):
        resp = client.post(
            "/v1/ops/upscale",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-up"


# ─── Job polling ─────────────────────────────────────────────────────────────


def test_get_job_queued_status(client: TestClient) -> None:
    mock_result = {
        "job_id": "job-123",
        "status": "queued",
        "result": None,
        "enqueue_time": "2024-01-01T12:00:00",
    }
    with patch(
        "mediakit.server.routes.jobs.get_job_result",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        resp = client.get("/v1/jobs/job-123")
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    assert resp.json()["result"] is None


def test_get_job_complete_with_result(client: TestClient) -> None:
    mock_result = {
        "job_id": "job-456",
        "status": "complete",
        "result": {"output": "/storage/outputs/result.webp", "seed": 42},
        "enqueue_time": "2024-01-01T12:00:00",
    }
    with patch(
        "mediakit.server.routes.jobs.get_job_result",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        resp = client.get("/v1/jobs/job-456")
    assert resp.status_code == 200
    assert resp.json()["result"]["seed"] == 42


def test_get_job_not_found_returns_404(client: TestClient) -> None:
    with patch(
        "mediakit.server.routes.jobs.get_job_result", new_callable=AsyncMock, return_value=None
    ):
        resp = client.get("/v1/jobs/nonexistent-id")
    assert resp.status_code == 404


# ─── Pipelines ───────────────────────────────────────────────────────────────


def test_pipeline_article_cover_enqueues(client: TestClient) -> None:
    with patch(
        "mediakit.server.routes.pipelines.enqueue",
        new_callable=AsyncMock,
        return_value="pipe-cover",
    ):
        resp = client.post(
            "/v1/pipelines/article-cover",
            data={"prompt": "abstract tech background", "slug": "test-post", "output_dir": "/tmp"},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "pipe-cover"
    assert resp.json()["status"] == "queued"


def test_pipeline_photo_finalize_enqueues(client: TestClient, jpeg_bytes: bytes) -> None:
    with patch(
        "mediakit.server.routes.pipelines.enqueue", new_callable=AsyncMock, return_value="pipe-fin"
    ):
        resp = client.post(
            "/v1/pipelines/photo-finalize",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"output_dir": "/tmp"},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "pipe-fin"


def test_pipeline_responsive_set_enqueues(client: TestClient, jpeg_bytes: bytes) -> None:
    with patch(
        "mediakit.server.routes.pipelines.enqueue", new_callable=AsyncMock, return_value="pipe-rs"
    ):
        resp = client.post(
            "/v1/pipelines/responsive-set",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"sizes": "640,1024", "formats": "webp"},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "pipe-rs"


# ─── product_shot pipeline ───────────────────────────────────────────────────


def test_pipeline_product_shot_enqueues(client: TestClient, jpeg_bytes: bytes) -> None:
    with patch(
        "mediakit.server.routes.pipelines.enqueue",
        new_callable=AsyncMock,
        return_value="pipe-ps",
    ):
        resp = client.post(
            "/v1/pipelines/product-shot",
            files={"file": ("product.jpg", jpeg_bytes, "image/jpeg")},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "pipe-ps"
    assert resp.json()["status"] == "queued"


def test_pipeline_product_shot_defaults(client: TestClient, jpeg_bytes: bytes) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "pipe-ps-defaults"

    with patch("mediakit.server.routes.pipelines.enqueue", side_effect=capture):
        client.post(
            "/v1/pipelines/product-shot",
            files={"file": ("product.jpg", jpeg_bytes, "image/jpeg")},
        )

    assert captured
    fn, params = captured[0]
    assert fn == "task_pipeline_product_shot"
    assert params["do_upscale"] is True
    assert params["bg_color"] == "#FFFFFF"
    assert params["padding_pct"] == 0.1
    assert params["quality"] == "high"
    assert params["formats"] is None
    assert params["widths"] is None


def test_pipeline_product_shot_custom_params(client: TestClient, jpeg_bytes: bytes) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "pipe-ps-custom"

    with patch("mediakit.server.routes.pipelines.enqueue", side_effect=capture):
        client.post(
            "/v1/pipelines/product-shot",
            files={"file": ("product.jpg", jpeg_bytes, "image/jpeg")},
            data={
                "bg_color": "#F5F5F5",
                "padding_pct": "0.15",
                "do_upscale": "false",
                "formats": "webp,avif",
                "widths": "640,1280",
            },
        )

    _, params = captured[0]
    assert params["bg_color"] == "#F5F5F5"
    assert params["padding_pct"] == 0.15
    assert params["do_upscale"] is False
    assert params["formats"] == ["webp", "avif"]
    assert params["widths"] == [640, 1280]


# ─── txt_to_video_hq pipeline ────────────────────────────────────────────────


def test_pipeline_txt_to_video_hq_enqueues(client: TestClient) -> None:
    with patch(
        "mediakit.server.routes.pipelines.enqueue",
        new_callable=AsyncMock,
        return_value="pipe-t2vhq",
    ):
        resp = client.post(
            "/v1/pipelines/txt-to-video-hq",
            data={"prompt": "a futuristic city at sunset", "output_dir": "/tmp/out"},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "pipe-t2vhq"
    assert resp.json()["status"] == "queued"


def test_pipeline_txt_to_video_hq_defaults(client: TestClient) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "pipe-t2vhq-defaults"

    with patch("mediakit.server.routes.pipelines.enqueue", side_effect=capture):
        client.post(
            "/v1/pipelines/txt-to-video-hq",
            data={"prompt": "test prompt", "output_dir": "/tmp/out"},
        )

    assert captured
    fn, params = captured[0]
    assert fn == "task_pipeline_txt_to_video_hq"
    assert params["img_backend"] == "sdxl"
    assert params["width"] == 768
    assert params["height"] == 512
    assert params["img_steps"] == 25
    assert params["vid_length"] == 49
    assert params["vid_seed"] == -1


def test_pipeline_txt_to_video_hq_flux_backend(client: TestClient) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "pipe-t2vhq-flux"

    with patch("mediakit.server.routes.pipelines.enqueue", side_effect=capture):
        client.post(
            "/v1/pipelines/txt-to-video-hq",
            data={
                "prompt": "neon cityscape",
                "output_dir": "/tmp/out",
                "img_backend": "flux",
                "width": "1024",
                "height": "576",
                "vid_steps": "20",
            },
        )

    _, params = captured[0]
    assert params["img_backend"] == "flux"
    assert params["width"] == 1024
    assert params["height"] == 576
    assert params["vid_steps"] == 20


# ─── photo_animate pipeline ──────────────────────────────────────────────────


def test_pipeline_photo_animate_enqueues(client: TestClient, jpeg_bytes: bytes) -> None:
    with patch(
        "mediakit.server.routes.pipelines.enqueue",
        new_callable=AsyncMock,
        return_value="pipe-anim",
    ):
        resp = client.post(
            "/v1/pipelines/photo-animate",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"prompt": "gentle breeze, hair flowing"},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "pipe-anim"
    assert resp.json()["status"] == "queued"


def test_pipeline_photo_animate_defaults(client: TestClient, jpeg_bytes: bytes) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "pipe-anim-defaults"

    with patch("mediakit.server.routes.pipelines.enqueue", side_effect=capture):
        client.post(
            "/v1/pipelines/photo-animate",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        )

    assert captured
    fn, params = captured[0]
    assert fn == "task_pipeline_photo_animate"
    assert params["remove_bg"] is False
    assert params["do_upscale"] is False
    assert params["width"] == 768
    assert params["height"] == 512
    assert params["length"] == 49
    assert params["fps"] == 24.0
    assert params["seed"] == -1


def test_pipeline_photo_animate_with_bg_remove(client: TestClient, jpeg_bytes: bytes) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "pipe-anim-bg"

    with patch("mediakit.server.routes.pipelines.enqueue", side_effect=capture):
        client.post(
            "/v1/pipelines/photo-animate",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={
                "prompt": "studio shoot",
                "remove_bg": "true",
                "do_upscale": "true",
                "upscale_scale": "3.0",
                "width": "1024",
                "height": "576",
                "seed": "42",
            },
        )

    assert captured
    _, params = captured[0]
    assert params["remove_bg"] is True
    assert params["do_upscale"] is True
    assert params["upscale_scale"] == 3.0
    assert params["width"] == 1024
    assert params["height"] == 576
    assert params["seed"] == 42


# ─── Video ops ───────────────────────────────────────────────────────────────


def test_txt2video_enqueues_job(client: TestClient) -> None:
    with patch(
        "mediakit.server.routes.jobs.enqueue", new_callable=AsyncMock, return_value="job-t2v"
    ):
        resp = client.post("/v1/ops/txt2video", data={"prompt": "Bangkok night market"})
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-t2v"
    assert resp.json()["status"] == "queued"


def test_txt2video_default_model_is_ltxv(client: TestClient) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "job-model"

    with patch("mediakit.server.routes.jobs.enqueue", side_effect=capture):
        client.post("/v1/ops/txt2video", data={"prompt": "test"})

    assert captured[0][1]["model"] == "ltxv"


def test_img2video_enqueues_job(client: TestClient, jpeg_bytes: bytes) -> None:
    with patch(
        "mediakit.server.routes.jobs.enqueue", new_callable=AsyncMock, return_value="job-i2v"
    ):
        resp = client.post(
            "/v1/ops/img2video",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"prompt": "person slowly looks to camera"},
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-i2v"


def test_txt2img_backend_flux_enqueues(client: TestClient) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "job-flux"

    with patch("mediakit.server.routes.jobs.enqueue", side_effect=capture):
        client.post("/v1/ops/txt2img", data={"prompt": "test", "backend": "flux"})

    assert captured[0][1]["backend"] == "flux"


def test_img_edit_backend_qwen_enqueues(client: TestClient, jpeg_bytes: bytes) -> None:
    captured: list[tuple] = []

    async def capture(fn: str, params: dict, **kw: object) -> str:  # type: ignore[misc]
        captured.append((fn, params))
        return "job-qwen"

    with patch("mediakit.server.routes.jobs.enqueue", side_effect=capture):
        client.post(
            "/v1/ops/img-edit",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"prompt": "white background", "backend": "qwen", "lora_strength": "0.8"},
        )

    assert captured[0][1]["backend"] == "qwen"
    assert captured[0][1]["lora_strength"] == 0.8


# ─── Output download endpoint ────────────────────────────────────────────────


def test_get_job_output_returns_file(client: TestClient, tmp_path: Path) -> None:  # noqa: F821
    import pathlib
    import tempfile

    # Create a real temp file to serve
    f = pathlib.Path(tempfile.mktemp(suffix=".jpg"))
    f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)  # minimal JPEG header
    try:
        mock_result = {
            "job_id": "job-out",
            "status": "complete",
            "result": {"output": str(f), "seed": 42},
            "enqueue_time": "2024-01-01T12:00:00",
        }
        with patch(
            "mediakit.server.routes.jobs.get_job_result",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.get("/v1/jobs/job-out/output")
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("attachment")
    finally:
        f.unlink(missing_ok=True)


def test_get_job_output_409_when_not_complete(client: TestClient) -> None:
    mock_result = {
        "job_id": "job-q",
        "status": "queued",
        "result": None,
        "enqueue_time": "2024-01-01T12:00:00",
    }
    with patch(
        "mediakit.server.routes.jobs.get_job_result",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        resp = client.get("/v1/jobs/job-q/output")
    assert resp.status_code == 409


def test_get_job_output_404_when_job_missing(client: TestClient) -> None:
    with patch(
        "mediakit.server.routes.jobs.get_job_result", new_callable=AsyncMock, return_value=None
    ):
        resp = client.get("/v1/jobs/gone/output")
    assert resp.status_code == 404


def test_get_job_output_410_when_file_deleted(client: TestClient) -> None:
    mock_result = {
        "job_id": "job-del",
        "status": "complete",
        "result": {"output": "/nonexistent/file.jpg"},
        "enqueue_time": "2024-01-01T12:00:00",
    }
    with patch(
        "mediakit.server.routes.jobs.get_job_result",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        resp = client.get("/v1/jobs/job-del/output")
    assert resp.status_code == 410
