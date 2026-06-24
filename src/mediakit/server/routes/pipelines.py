"""Pipeline endpoints — compositions of ops."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import BaseModel

from mediakit.jobs.queue import enqueue
from mediakit.server.deps import require_token
from mediakit.server.utils import save_upload

router = APIRouter(prefix="/v1/pipelines", dependencies=[Depends(require_token)])


class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"


@router.post("/article-cover", status_code=status.HTTP_202_ACCEPTED)
async def pipeline_article_cover(
    prompt: Annotated[str, Form()],
    slug: Annotated[str, Form()],
    output_dir: Annotated[str, Form()],
    negative: Annotated[str, Form()] = "",
    backend: Annotated[str, Form()] = "sdxl",
    steps: Annotated[int, Form()] = 25,
    cfg: Annotated[float, Form()] = 7.5,
    seed: Annotated[int, Form()] = -1,
    responsive: Annotated[str, Form()] = "",  # comma-separated widths or empty
) -> JobAccepted:
    widths = [int(x) for x in responsive.split(",") if x.strip()] or None
    job_id = await enqueue(
        "task_pipeline_article_cover",
        {
            "prompt": prompt,
            "slug": slug,
            "output_dir": output_dir,
            "negative_prompt": negative,
            "backend": backend,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "responsive_widths": widths,
        },
    )
    return JobAccepted(job_id=job_id)


@router.post("/photo-finalize", status_code=status.HTTP_202_ACCEPTED)
async def pipeline_photo_finalize(
    file: Annotated[UploadFile, File()],
    output_dir: Annotated[str, Form()],
    background_mode: Annotated[str, Form()] = "transparent",
    upscale_scale: Annotated[float, Form()] = 2.0,
) -> JobAccepted:
    src = save_upload(file)
    job_id = await enqueue(
        "task_pipeline_photo_finalize",
        {
            "input": src,
            "output_dir": output_dir,
            "background_mode": background_mode,
            "upscale_scale": upscale_scale,
        },
    )
    return JobAccepted(job_id=job_id)


@router.post("/product-shot", status_code=status.HTTP_202_ACCEPTED)
async def pipeline_product_shot(
    file: Annotated[UploadFile, File()],
    output_dir: Annotated[str, Form()] = "",
    birefnet_model: Annotated[str, Form()] = "hr",
    bg_color: Annotated[str, Form()] = "#FFFFFF",
    padding_pct: Annotated[float, Form()] = 0.1,
    do_upscale: Annotated[bool, Form()] = True,
    upscale_model: Annotated[str, Form()] = "nmkd",
    upscale_scale: Annotated[float, Form()] = 2.0,
    formats: Annotated[str, Form()] = "",  # comma-separated, empty = webp
    widths: Annotated[str, Form()] = "",  # comma-separated, empty = defaults
    quality: Annotated[str, Form()] = "high",
) -> JobAccepted:
    src = save_upload(file)
    fmts = [x.strip() for x in formats.split(",") if x.strip()] or None
    wids = [int(x) for x in widths.split(",") if x.strip()] or None
    job_id = await enqueue(
        "task_pipeline_product_shot",
        {
            "input": str(src),
            "output_dir": output_dir or None,
            "birefnet_model": birefnet_model,
            "bg_color": bg_color,
            "padding_pct": padding_pct,
            "do_upscale": do_upscale,
            "upscale_model": upscale_model,
            "upscale_scale": upscale_scale,
            "formats": fmts,
            "widths": wids,
            "quality": quality,
        },
    )
    return JobAccepted(job_id=job_id)


@router.post("/txt-to-video-hq", status_code=status.HTTP_202_ACCEPTED)
async def pipeline_txt_to_video_hq(
    prompt: Annotated[str, Form()],
    output_dir: Annotated[str, Form()],
    negative_prompt: Annotated[str, Form()] = "",
    img_backend: Annotated[str, Form()] = "sdxl",
    width: Annotated[int, Form()] = 768,
    height: Annotated[int, Form()] = 512,
    img_steps: Annotated[int, Form()] = 25,
    img_cfg: Annotated[float, Form()] = 7.5,
    img_seed: Annotated[int, Form()] = -1,
    vid_length: Annotated[int, Form()] = 49,
    vid_fps: Annotated[float, Form()] = 24.0,
    vid_steps: Annotated[int, Form()] = 15,
    vid_cfg: Annotated[float, Form()] = 2.0,
    vid_seed: Annotated[int, Form()] = -1,
) -> JobAccepted:
    job_id = await enqueue(
        "task_pipeline_txt_to_video_hq",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_dir": output_dir,
            "img_backend": img_backend,
            "width": width,
            "height": height,
            "img_steps": img_steps,
            "img_cfg": img_cfg,
            "img_seed": img_seed,
            "vid_length": vid_length,
            "vid_fps": vid_fps,
            "vid_steps": vid_steps,
            "vid_cfg": vid_cfg,
            "vid_seed": vid_seed,
        },
    )
    return JobAccepted(job_id=job_id)


@router.post("/seamless-video", status_code=status.HTTP_202_ACCEPTED)
async def pipeline_seamless_video(
    prompt: Annotated[str, Form()],
    output_dir: Annotated[str, Form()],
    file: Annotated[UploadFile | None, File()] = None,  # optional first-frame image
    negative_prompt: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "cogvideox",
    total_frames: Annotated[int, Form()] = 97,
    fps: Annotated[float | None, Form()] = None,
    width: Annotated[int | None, Form()] = None,
    height: Annotated[int | None, Form()] = None,
    overlap_frames: Annotated[int, Form()] = 8,
    steps: Annotated[int, Form()] = 50,
    cfg: Annotated[float, Form()] = 6.0,
    seed: Annotated[int, Form()] = -1,
) -> JobAccepted:
    src = save_upload(file) if file is not None else None
    job_id = await enqueue(
        "task_pipeline_seamless_video",
        {
            "prompt": prompt,
            "output_dir": output_dir,
            "input": str(src) if src is not None else None,
            "negative_prompt": negative_prompt,
            "model": model,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "overlap_frames": overlap_frames,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
        },
    )
    return JobAccepted(job_id=job_id)


@router.post("/photo-animate", status_code=status.HTTP_202_ACCEPTED)
async def pipeline_photo_animate(
    file: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()] = "",
    negative_prompt: Annotated[str, Form()] = "",
    output_dir: Annotated[str, Form()] = "",
    remove_bg: Annotated[bool, Form()] = False,
    birefnet_model: Annotated[str, Form()] = "hr",
    do_upscale: Annotated[bool, Form()] = False,
    upscale_model: Annotated[str, Form()] = "nmkd",
    upscale_scale: Annotated[float, Form()] = 2.0,
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
        "task_pipeline_photo_animate",
        {
            "input": str(src),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_dir": output_dir or None,
            "remove_bg": remove_bg,
            "birefnet_model": birefnet_model,
            "do_upscale": do_upscale,
            "upscale_model": upscale_model,
            "upscale_scale": upscale_scale,
            "width": width,
            "height": height,
            "length": length,
            "fps": fps,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
        },
    )
    return JobAccepted(job_id=job_id)


@router.post("/responsive-set", status_code=status.HTTP_202_ACCEPTED)
async def pipeline_responsive_set(
    file: Annotated[UploadFile, File()],
    output_dir: Annotated[str, Form()] = "",
    sizes: Annotated[str, Form()] = "",  # comma-separated widths, empty = defaults
    formats: Annotated[str, Form()] = "",  # comma-separated formats, empty = webp+avif
    quality: Annotated[str, Form()] = "high",
    generate_lqip: Annotated[bool, Form()] = True,
) -> JobAccepted:
    src = save_upload(file)
    widths = [int(x) for x in sizes.split(",") if x.strip()] or None
    fmts = [x.strip() for x in formats.split(",") if x.strip()] or None
    job_id = await enqueue(
        "task_pipeline_responsive_set",
        {
            "input": str(src),
            "output_dir": output_dir or None,
            "widths": widths,
            "formats": fmts,
            "quality": quality,
            "generate_lqip": generate_lqip,
        },
    )
    return JobAccepted(job_id=job_id)
