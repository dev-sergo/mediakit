"""photo_animate pipeline.

[bg_remove →] [upscale →] img2video

Animates a photo into a short video clip. Background removal and upscaling
are optional pre-processing steps before animation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from mediakit.ops.bg_remove import bg_remove
from mediakit.ops.img2video import img2video
from mediakit.ops.upscale import upscale
from mediakit.pipelines.base import BasePipeline, PipelineResult
from mediakit.schemas.ai_ops import BgRemoveParams, BiRefNetModel, UpscaleModel, UpscaleParams
from mediakit.schemas.video_ops import Img2VideoParams

log = structlog.get_logger(__name__)


class PhotoAnimatePipeline(BasePipeline):
    name = "photo_animate"

    async def run(  # type: ignore[override]
        self,
        *,
        input: Path,
        prompt: str = "",
        negative_prompt: str = "",
        output_dir: Path | None = None,
        remove_bg: bool = False,
        birefnet_model: BiRefNetModel = BiRefNetModel.hr,
        do_upscale: bool = False,
        upscale_model: UpscaleModel = UpscaleModel.nmkd,
        upscale_scale: float = 2.0,
        width: int = 768,
        height: int = 512,
        length: int = 49,
        fps: float = 24.0,
        steps: int = 15,
        cfg: float = 2.0,
        seed: int = -1,
        **_: Any,
    ) -> PipelineResult:
        out_dir = output_dir or input.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input.stem
        outputs: list[Path] = []

        current = input

        # 1. Optional background removal
        if remove_bg:
            log.info("photo_animate.bg_remove", input=str(current))
            bg = await bg_remove(BgRemoveParams(
                input=current,
                output=out_dir / f"{stem}_nobg.png",
                model=birefnet_model,
            ))
            current = bg.output
            outputs.append(current)

        # 2. Optional upscale
        if do_upscale:
            log.info("photo_animate.upscale", scale=upscale_scale)
            up = await upscale(UpscaleParams(
                input=current,
                output=out_dir / f"{stem}_upscaled.png",
                model=upscale_model,
                scale=upscale_scale,
            ))
            current = up.output
            outputs.append(current)

        # 3. Animate
        log.info("photo_animate.img2video", input=str(current), length=length)
        vid = await img2video(Img2VideoParams(
            input=current,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output=out_dir / f"{stem}_animated.mp4",
            width=width,
            height=height,
            length=length,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
        ))
        outputs.append(vid.output)

        log.info("photo_animate.done", files=len(outputs), seed=vid.seed, duration_s=vid.duration_s)
        return PipelineResult(
            outputs=outputs,
            meta={"seed": vid.seed, "duration_s": vid.duration_s, "stem": stem},
        )
