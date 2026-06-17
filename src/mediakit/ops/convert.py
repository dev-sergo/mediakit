import asyncio

import structlog
from PIL import Image

from mediakit.backends.native import encoder
from mediakit.schemas.ops import ConvertParams, ConvertResult

log = structlog.get_logger(__name__)


def _run(params: ConvertParams) -> ConvertResult:
    input_bytes = params.input.stat().st_size
    img = Image.open(params.input)
    output = encoder.resolve_output(params.input, params.format, params.output)
    encoder.encode(img, output, params.format, params.quality_int())
    output_bytes = output.stat().st_size
    log.info("convert", input=str(params.input), output=str(output))
    return ConvertResult(output=output, input_bytes=input_bytes, output_bytes=output_bytes)


async def convert(params: ConvertParams) -> ConvertResult:
    return await asyncio.to_thread(_run, params)
