"""Async AI ops — txt2img, img_edit, bg_remove, upscale.

POST enqueues the job → returns 202 + job_id.
GET  /v1/jobs/{id} → polls status + result when done.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from mediakit.jobs.queue import enqueue, get_job_result
from mediakit.schemas.ai_ops import BiRefNetModel, UpscaleModel
from mediakit.server.deps import require_token
from mediakit.server.utils import save_upload

router = APIRouter(dependencies=[Depends(require_token)])


class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"


# ─── Enqueue endpoints ────────────────────────────────────────────────────────

@router.post("/v1/ops/txt2img", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_txt2img(
    prompt: Annotated[str, Form()],
    negative: Annotated[str, Form()] = "",
    backend: Annotated[str, Form()] = "sdxl",
    width: Annotated[int, Form()] = 1024,
    height: Annotated[int, Form()] = 1024,
    steps: Annotated[int, Form()] = 25,
    cfg: Annotated[float, Form()] = 7.5,
    seed: Annotated[int, Form()] = -1,
    checkpoint: Annotated[str, Form()] = "RealVisXL_V5.0_inpainting.safetensors",
) -> JobAccepted:
    import secrets as _sec
    actual_seed = _sec.randbits(32) if seed == -1 else seed
    job_id = await enqueue(
        "task_txt2img",
        {"prompt": prompt, "negative_prompt": negative, "backend": backend,
         "width": width, "height": height,
         "steps": steps, "cfg": cfg, "seed": actual_seed, "checkpoint": checkpoint},
    )
    return JobAccepted(job_id=job_id)


@router.post("/v1/ops/img-edit", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_img_edit(
    file: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()],
    negative: Annotated[str, Form()] = "",
    backend: Annotated[str, Form()] = "sdxl",
    width: Annotated[int, Form()] = 1024,
    height: Annotated[int, Form()] = 1024,
    steps: Annotated[int, Form()] = 25,
    cfg: Annotated[float, Form()] = 7.5,
    seed: Annotated[int, Form()] = -1,
    mask: Annotated[str, Form()] = "background",
    lora_strength: Annotated[float, Form()] = 1.0,
) -> JobAccepted:
    import secrets as _sec
    src = save_upload(file)
    actual_seed = _sec.randbits(32) if seed == -1 else seed
    job_id = await enqueue(
        "task_img_edit",
        {"input": src, "prompt": prompt, "negative_prompt": negative,
         "backend": backend, "width": width, "height": height,
         "steps": steps, "cfg": cfg, "seed": actual_seed,
         "mask_target": mask, "lora_strength": lora_strength},
    )
    return JobAccepted(job_id=job_id)


@router.post("/v1/ops/bg-remove", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_bg_remove(
    file: Annotated[UploadFile, File()],
    model: Annotated[BiRefNetModel, Form()] = BiRefNetModel.hr,
    background: Annotated[str, Form()] = "transparent",
    color: Annotated[str, Form()] = "#FFFFFF",
) -> JobAccepted:
    src = save_upload(file)
    job_id = await enqueue(
        "task_bg_remove",
        {
            "input": src, "model": model.value,
            "background_mode": background, "background_color": color,
        },
    )
    return JobAccepted(job_id=job_id)


@router.post("/v1/ops/upscale", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_upscale(
    file: Annotated[UploadFile, File()],
    model: Annotated[UpscaleModel, Form()] = UpscaleModel.nmkd,
    scale: Annotated[float, Form()] = 2.0,
) -> JobAccepted:
    src = save_upload(file)
    job_id = await enqueue(
        "task_upscale",
        {"input": src, "model": model.value, "scale": scale},
    )
    return JobAccepted(job_id=job_id)


# ─── Video ops ───────────────────────────────────────────────────────────────

@router.post("/v1/ops/txt2video", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_txt2video(
    prompt: Annotated[str, Form()],
    negative: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "ltxv",
    width: Annotated[int, Form()] = 768,
    height: Annotated[int, Form()] = 512,
    length: Annotated[int, Form()] = 49,
    fps: Annotated[float, Form()] = 24.0,
    steps: Annotated[int, Form()] = 15,
    cfg: Annotated[float, Form()] = 2.5,
    seed: Annotated[int, Form()] = -1,
) -> JobAccepted:
    job_id = await enqueue(
        "task_txt2video",
        {"prompt": prompt, "negative_prompt": negative, "model": model,
         "width": width, "height": height, "length": length, "fps": fps,
         "steps": steps, "cfg": cfg, "seed": seed},
    )
    return JobAccepted(job_id=job_id)


@router.post("/v1/ops/img2video", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_img2video(
    file: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()],
    negative: Annotated[str, Form()] = "",
    width: Annotated[int, Form()] = 768,
    height: Annotated[int, Form()] = 512,
    length: Annotated[int, Form()] = 49,
    fps: Annotated[float, Form()] = 24.0,
    steps: Annotated[int, Form()] = 15,
    cfg: Annotated[float, Form()] = 2.0,
    seed: Annotated[int, Form()] = -1,
) -> JobAccepted:
    src = save_upload(file)
    job_id = await enqueue(
        "task_img2video",
        {"input": str(src), "prompt": prompt, "negative_prompt": negative,
         "width": width, "height": height, "length": length, "fps": fps,
         "steps": steps, "cfg": cfg, "seed": seed},
    )
    return JobAccepted(job_id=job_id)


# ─── Job status polling ───────────────────────────────────────────────────────

@router.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    result = await get_job_result(job_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return result


@router.get("/v1/jobs/{job_id}/output")
async def get_job_output(job_id: str) -> FileResponse:
    """Download the primary output file of a completed job."""
    job = await get_job_result(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job["status"] != "complete":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job not complete yet: {job['status']}",
        )

    result = job.get("result") or {}
    if "output" in result:
        file_path = Path(str(result["output"]))
    elif "outputs" in result and result["outputs"]:
        file_path = Path(str(result["outputs"][0]))
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No output in job result")

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Output file expired or was deleted by cleanup",
        )

    return FileResponse(path=str(file_path), filename=file_path.name)
