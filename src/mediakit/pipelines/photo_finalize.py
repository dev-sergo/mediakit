"""photo_finalize pipeline.

bg_remove → upscale → compress → [variants]

Suitable for product photos: cut out the subject, upscale for marketplace
resolution, compress, and optionally generate responsive variants.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from mediakit.ops.bg_remove import bg_remove
from mediakit.ops.compress import compress
from mediakit.ops.upscale import upscale
from mediakit.ops.variants import variants as make_variants
from mediakit.pipelines.base import BasePipeline, PipelineResult
from mediakit.schemas.ai_ops import BgRemoveParams, BiRefNetModel, UpscaleModel, UpscaleParams
from mediakit.schemas.ops import CompressParams, ImageFormat, Quality, VariantsParams

log = structlog.get_logger(__name__)


class PhotoFinalizePipeline(BasePipeline):
    name = "photo_finalize"

    async def run(  # type: ignore[override]
        self,
        *,
        input: Path,
        output_dir: Path | None = None,
        birefnet_model: BiRefNetModel = BiRefNetModel.hr,
        background_mode: str = "transparent",
        background_color: str = "#FFFFFF",
        upscale_model: UpscaleModel = UpscaleModel.nmkd,
        upscale_scale: float = 2.0,
        compress_format: ImageFormat = ImageFormat.webp,
        quality: Quality = Quality.high,
        responsive_widths: list[int] | None = None,
        **_: Any,
    ) -> PipelineResult:
        out_dir = output_dir or input.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input.stem

        # 1. Background removal
        log.info("photo_finalize.bg_remove", input=str(input))
        bg = await bg_remove(BgRemoveParams(
            input=input,
            output=out_dir / f"{stem}_nobg.png",
            model=birefnet_model,
            background_mode=background_mode,  # type: ignore[arg-type]
            background_color=background_color,
        ))

        # 2. Upscale
        log.info("photo_finalize.upscale", scale=upscale_scale)
        up = await upscale(UpscaleParams(
            input=bg.output,
            output=out_dir / f"{stem}_upscaled.png",
            model=upscale_model,
            scale=upscale_scale,
        ))

        # 3. Compress
        final_ext = compress_format.extension
        final = out_dir / f"{stem}_final.{final_ext}"
        await compress(
            CompressParams(input=up.output, output=final, format=compress_format, quality=quality)
        )
        up.output.unlink(missing_ok=True)
        up.output.with_name(up.output.name + ".json").unlink(missing_ok=True)

        outputs = [bg.output, final]

        # 4. Optional responsive variants
        if responsive_widths:
            vresult = await make_variants(VariantsParams(
                input=final,
                output_dir=out_dir,
                widths=responsive_widths,
                formats=[compress_format],
                quality=quality,
                stem=stem,
            ))
            outputs += [v.path for v in vresult.variants]

        log.info("photo_finalize.done", files=len(outputs))
        return PipelineResult(outputs=outputs)
