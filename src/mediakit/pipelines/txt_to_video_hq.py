"""txt_to_video_hq pipeline.

txt2img → img2video

Generates a keyframe from a text prompt, then animates it. Produces higher
quality and more compositionally consistent video than direct txt2video because
the keyframe can be inspected/swapped independently before animation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import structlog

from mediakit.ops.img2video import img2video
from mediakit.ops.txt2img import txt2img
from mediakit.pipelines.base import BasePipeline, PipelineResult
from mediakit.schemas.ai_ops import Txt2ImgParams
from mediakit.schemas.video_ops import Img2VideoParams

log = structlog.get_logger(__name__)


class TxtToVideoHqPipeline(BasePipeline):
    name = "txt_to_video_hq"

    async def run(  # type: ignore[override]
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        output_dir: Path,
        img_backend: Literal["sdxl", "flux"] = "sdxl",
        width: int = 768,
        height: int = 512,
        img_steps: int = 25,
        img_cfg: float = 7.5,
        img_seed: int = -1,
        vid_length: int = 49,
        vid_fps: float = 24.0,
        vid_steps: int = 15,
        vid_cfg: float = 2.0,
        vid_seed: int = -1,
        **_: Any,
    ) -> PipelineResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate keyframe
        log.info(
            "txt_to_video_hq.txt2img",
            backend=img_backend,
            width=width,
            height=height,
        )
        frame = await txt2img(
            Txt2ImgParams(
                prompt=prompt,
                negative_prompt=negative_prompt,
                backend=img_backend,
                output=output_dir / "keyframe.png",
                width=width,
                height=height,
                steps=img_steps,
                cfg=img_cfg,
                seed=img_seed,
            )
        )
        outputs: list[Path] = [frame.output]

        # 2. Animate keyframe — reuse the prompt for motion guidance
        log.info("txt_to_video_hq.img2video", img_seed=frame.seed)
        vid = await img2video(
            Img2VideoParams(
                input=frame.output,
                prompt=prompt,
                negative_prompt=negative_prompt,
                output=output_dir / "video.mp4",
                width=width,
                height=height,
                length=vid_length,
                fps=vid_fps,
                steps=vid_steps,
                cfg=vid_cfg,
                seed=vid_seed,
            )
        )
        outputs.append(vid.output)

        log.info(
            "txt_to_video_hq.done",
            img_seed=frame.seed,
            vid_seed=vid.seed,
            duration_s=vid.duration_s,
        )
        return PipelineResult(
            outputs=outputs,
            meta={
                "img_seed": frame.seed,
                "vid_seed": vid.seed,
                "duration_s": vid.duration_s,
            },
        )
