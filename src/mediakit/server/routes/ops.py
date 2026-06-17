"""Sync ops — compress, resize, convert, variants, lqip.

These run in-process (Pillow, no GPU) and return immediately.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from mediakit.ops import compress, convert, lqip, resize, variants
from mediakit.schemas.ops import (
    CompressParams,
    ConvertParams,
    ImageFormat,
    LqipParams,
    Quality,
    ResizeMode,
    ResizeParams,
    VariantsParams,
    parse_quality,
)
from mediakit.server.deps import require_token
from mediakit.server.utils import save_upload

router = APIRouter(prefix="/v1/ops", dependencies=[Depends(require_token)])


class FileResult(BaseModel):
    output_path: str
    input_bytes: int | None = None
    output_bytes: int | None = None
    savings_pct: float | None = None


@router.post("/compress")
async def compress_op(
    file: Annotated[UploadFile, File()],
    format: Annotated[ImageFormat, Form()] = ImageFormat.webp,
    quality: Annotated[str, Form()] = "high",
    max_width: Annotated[int | None, Form()] = None,
) -> FileResult:
    src = save_upload(file)
    q: Quality | int = parse_quality(quality)
    result = await compress(
        CompressParams(input=src, format=format, quality=q, max_width=max_width)
    )
    return FileResult(
        output_path=str(result.output),
        input_bytes=result.input_bytes,
        output_bytes=result.output_bytes,
        savings_pct=result.savings_pct,
    )


@router.post("/resize")
async def resize_op(
    file: Annotated[UploadFile, File()],
    width: Annotated[int, Form()],
    height: Annotated[int, Form()],
    mode: Annotated[ResizeMode, Form()] = ResizeMode.fit,
) -> FileResult:
    src = save_upload(file)
    result = await resize(ResizeParams(input=src, width=width, height=height, mode=mode))
    return FileResult(output_path=str(result.output))


@router.post("/convert")
async def convert_op(
    file: Annotated[UploadFile, File()],
    format: Annotated[ImageFormat, Form()],
    quality: Annotated[str, Form()] = "high",
) -> FileResult:
    src = save_upload(file)
    q: Quality | int = parse_quality(quality)
    result = await convert(ConvertParams(input=src, format=format, quality=q))
    return FileResult(
        output_path=str(result.output),
        input_bytes=result.input_bytes,
        output_bytes=result.output_bytes,
    )


@router.post("/lqip")
async def lqip_op(
    file: Annotated[UploadFile, File()],
    size: Annotated[int, Form()] = 16,
) -> dict[str, Any]:
    src = save_upload(file)
    result = await lqip(LqipParams(input=src, size=size))
    return {"data_url": result.data_url, "width": result.width, "height": result.height}


@router.post("/variants")
async def variants_op(
    file: Annotated[UploadFile, File()],
    sizes: Annotated[str, Form()] = "640,768,1024,1280,1536",
    formats: Annotated[str, Form()] = "webp",
    quality: Annotated[str, Form()] = "high",
) -> dict[str, Any]:
    src = save_upload(file)
    widths = [int(x.strip()) for x in sizes.split(",")]
    fmts = [ImageFormat(x.strip()) for x in formats.split(",")]
    q: Quality | int = parse_quality(quality)
    result = await variants(VariantsParams(input=src, widths=widths, formats=fmts, quality=q))
    return {
        "variants": [
            {"path": str(v.path), "width": v.width, "height": v.height, "format": v.format}
            for v in result.variants
        ]
    }
