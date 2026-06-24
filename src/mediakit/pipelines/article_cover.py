"""article_cover pipeline — txt2img → smart_crop(1200×630) → compress → [variants]."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from mediakit.ops.compress import compress
from mediakit.ops.resize import resize
from mediakit.ops.txt2img import txt2img
from mediakit.ops.variants import variants as make_variants
from mediakit.pipelines.base import BasePipeline, PipelineResult
from mediakit.schemas.ai_ops import Txt2ImgParams
from mediakit.schemas.ops import (
    CompressParams,
    ImageFormat,
    Quality,
    ResizeMode,
    ResizeParams,
    VariantsParams,
)

log = structlog.get_logger(__name__)

_COVER_W = 1200
_COVER_H = 630


class ArticleCoverPipeline(BasePipeline):
    name = "article_cover"

    async def run(  # type: ignore[override]
        self,
        *,
        prompt: str,
        output_dir: Path,
        slug: str,
        negative_prompt: str = "",
        gen_width: int = 1024,
        gen_height: int = 1024,
        steps: int = 25,
        cfg: float = 7.5,
        seed: int = -1,
        checkpoint: str = "RealVisXL_V5.0_inpainting.safetensors",
        backend: str = "sdxl",  # "sdxl" | "flux" — flux gives better quality
        cover_format: ImageFormat = ImageFormat.jpeg,
        responsive_widths: list[int] | None = None,
        **_: Any,
    ) -> PipelineResult:
        import secrets

        actual_seed = secrets.randbits(32) if seed == -1 else seed
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate raw image
        log.info("article_cover.generate", slug=slug, seed=actual_seed, backend=backend)
        gen = await txt2img(
            Txt2ImgParams(
                prompt=prompt,
                negative_prompt=negative_prompt,
                backend=backend,  # type: ignore[arg-type]
                checkpoint=checkpoint,
                width=gen_width,
                height=gen_height,
                steps=steps,
                cfg=cfg,
                seed=actual_seed,
            )
        )

        # 2. Smart-crop to 1200×630
        cover_raw = output_dir / f"{slug}_raw.png"
        await resize(
            ResizeParams(
                input=gen.output,
                output=cover_raw,
                width=_COVER_W,
                height=_COVER_H,
                mode=ResizeMode.smart_crop,
            )
        )

        # 3. Compress to requested format
        cover = output_dir / f"cover.{cover_format.extension}"
        await compress(
            CompressParams(
                input=cover_raw,
                output=cover,
                format=cover_format,
                quality=Quality.high,
            )
        )
        cover_raw.unlink(missing_ok=True)

        outputs = [cover]

        # 4. Optional responsive variants (webp)
        if responsive_widths:
            vresult = await make_variants(
                VariantsParams(
                    input=cover,
                    output_dir=output_dir,
                    widths=responsive_widths,
                    formats=[ImageFormat.webp],
                    quality=Quality.high,
                    stem="cover",
                )
            )
            outputs += [v.path for v in vresult.variants]

        log.info("article_cover.done", slug=slug, files=len(outputs))
        return PipelineResult(outputs=outputs, meta={"seed": actual_seed, "slug": slug})
