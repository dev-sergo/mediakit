"""responsive_set pipeline — compress → variants(webp+avif) → lqip."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import structlog

from mediakit.ops.compress import compress
from mediakit.ops.lqip import lqip
from mediakit.ops.variants import variants as make_variants
from mediakit.pipelines.base import BasePipeline, PipelineResult
from mediakit.schemas.ops import CompressParams, ImageFormat, LqipParams, Quality, VariantsParams

log = structlog.get_logger(__name__)

_DEFAULT_WIDTHS = [640, 768, 1024, 1280, 1536]
_DEFAULT_FORMATS = [ImageFormat.webp, ImageFormat.avif]


class ResponsiveSetPipeline(BasePipeline):
    name = "responsive_set"

    async def run(  # type: ignore[override]
        self,
        *,
        input: Path,
        output_dir: Path | None = None,
        widths: list[int] | None = None,
        formats: list[ImageFormat] | None = None,
        quality: Quality = Quality.high,
        generate_lqip: bool = True,
        **_: Any,
    ) -> PipelineResult:
        out_dir = output_dir or input.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        widths = widths or _DEFAULT_WIDTHS
        formats = formats or _DEFAULT_FORMATS

        with tempfile.TemporaryDirectory() as _tmp:
            # 1. Compress to a temp location — variants and lqip use it, then it's discarded
            compressed = Path(_tmp) / f"{input.stem}_compressed{input.suffix}"
            await compress(CompressParams(
                input=input, output=compressed, format=ImageFormat.jpeg, quality=quality
            ))

            # 2. Responsive variants (written to out_dir, not tmp)
            vresult = await make_variants(VariantsParams(
                input=compressed,
                output_dir=out_dir,
                widths=widths,
                formats=formats,
                quality=quality,
                stem=input.stem,
            ))

            # 3. LQIP
            lqip_path: Path | None = None
            if generate_lqip:
                lresult = await lqip(LqipParams(input=compressed))
                lqip_path = out_dir / f"{input.stem}.lqip"
                lqip_path.write_text(lresult.data_url)

        outputs: list[Path] = [v.path for v in vresult.variants]
        if lqip_path is not None:
            outputs.append(lqip_path)

        log.info("responsive_set.done", input=str(input), files=len(outputs))
        return PipelineResult(outputs=outputs)
