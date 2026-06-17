import asyncio
import base64

import structlog
from PIL import Image

from mediakit.backends.native.encoder import encode_bytes
from mediakit.schemas.ops import ImageFormat, LqipParams, LqipResult

log = structlog.get_logger(__name__)


def _run(params: LqipParams) -> LqipResult:
    img = Image.open(params.input).convert("RGB")
    src_w, src_h = img.size

    if src_w >= src_h:
        new_w, new_h = params.size, max(1, round(src_h * params.size / src_w))
    else:
        new_w, new_h = max(1, round(src_w * params.size / src_h)), params.size

    tiny = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    raw = encode_bytes(tiny, ImageFormat.webp, 20)
    data_url = "data:image/webp;base64," + base64.b64encode(raw).decode()
    return LqipResult(data_url=data_url, width=new_w, height=new_h)


async def lqip(params: LqipParams) -> LqipResult:
    return await asyncio.to_thread(_run, params)
