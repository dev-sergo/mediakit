import asyncio

import structlog
from PIL import Image

from mediakit.backends.native import encoder
from mediakit.schemas.ops import CompressParams, CompressResult

log = structlog.get_logger(__name__)


def _run(params: CompressParams) -> CompressResult:
    input_bytes = params.input.stat().st_size
    img = Image.open(params.input)

    if params.max_width and img.width > params.max_width:
        ratio = params.max_width / img.width
        img = img.resize(
            (params.max_width, round(img.height * ratio)),
            Image.Resampling.LANCZOS,
        )

    if params.strip_metadata:
        img = encoder.strip_metadata(img)

    output = encoder.resolve_output(params.input, params.format, params.output)
    encoder.encode(img, output, params.format, params.quality_int())

    output_bytes = output.stat().st_size
    savings = (1 - output_bytes / input_bytes) * 100 if input_bytes > 0 else 0.0
    log.info(
        "compress",
        input=str(params.input),
        output=str(output),
        input_kb=round(input_bytes / 1024, 1),
        output_kb=round(output_bytes / 1024, 1),
        savings_pct=round(savings, 1),
    )
    return CompressResult(
        output=output,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        savings_pct=round(savings, 2),
    )


async def compress(params: CompressParams) -> CompressResult:
    return await asyncio.to_thread(_run, params)
