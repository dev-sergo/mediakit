import asyncio

import structlog
from PIL import Image

from mediakit.backends.native import encoder, resizer
from mediakit.schemas.ops import ImageFormat, ResizeParams, ResizeResult

log = structlog.get_logger(__name__)

_PIL_TO_FORMAT: dict[str, ImageFormat] = {
    "JPEG": ImageFormat.jpeg,
    "WEBP": ImageFormat.webp,
    "AVIF": ImageFormat.avif,
    "PNG": ImageFormat.png,
}


def _run(params: ResizeParams) -> ResizeResult:
    img = Image.open(params.input)

    img = resizer.resize(img, params.width, params.height, params.mode, params.pad_color)

    fmt = params.format or _PIL_TO_FORMAT.get(img.format or "", ImageFormat.webp)
    output = encoder.resolve_output(params.input, fmt, params.output)
    encoder.encode(img, output, fmt, 85)

    log.info(
        "resize", input=str(params.input), output=str(output), size=f"{img.width}x{img.height}"
    )
    return ResizeResult(output=output, width=img.width, height=img.height)


async def resize(params: ResizeParams) -> ResizeResult:
    return await asyncio.to_thread(_run, params)
