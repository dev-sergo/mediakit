import asyncio

import structlog
from PIL import Image

from mediakit.backends.native import encoder
from mediakit.schemas.ops import VariantFile, VariantsParams, VariantsResult

log = structlog.get_logger(__name__)


def _run(params: VariantsParams) -> VariantsResult:
    img = Image.open(params.input)
    src_w, src_h = img.size
    stem = params.stem or params.input.stem
    out_dir = params.output_dir or params.input.parent
    q = params.quality_int()
    variants: list[VariantFile] = []

    for width in params.widths:
        if width > src_w:
            continue  # never upscale
        ratio = width / src_w
        height = round(src_h * ratio)
        resized = img.copy()
        resized.thumbnail((width, height), Image.Resampling.LANCZOS)

        for fmt in params.formats:
            filename = f"{stem}-{width}w.{fmt.extension}"
            out_path = out_dir / filename
            encoder.encode(resized, out_path, fmt, q)
            variants.append(
                VariantFile(path=out_path, width=resized.width, height=resized.height, format=fmt)
            )

    log.info("variants", input=str(params.input), count=len(variants))
    return VariantsResult(variants=variants)


async def variants(params: VariantsParams) -> VariantsResult:
    return await asyncio.to_thread(_run, params)
