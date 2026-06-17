"""Unit tests for native (Pillow-based) ops — no ComfyUI or Redis needed."""
import io
from pathlib import Path

import pytest
from PIL import Image

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
)


@pytest.fixture()
def sample_jpg(tmp_path: Path) -> Path:
    img = Image.new("RGB", (1200, 630), color=(200, 100, 50))
    path = tmp_path / "sample.jpg"
    img.save(path, format="JPEG", quality=90)
    return path


@pytest.fixture()
def sample_png(tmp_path: Path) -> Path:
    img = Image.new("RGBA", (800, 600), color=(0, 128, 255, 200))
    path = tmp_path / "sample.png"
    img.save(path, format="PNG")
    return path


# ─── compress ────────────────────────────────────────────────────────────────

async def test_compress_jpeg_to_webp(sample_jpg: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.webp"
    result = await compress(CompressParams(input=sample_jpg, output=out, format=ImageFormat.webp))
    assert result.output == out
    assert out.exists()
    assert result.output_bytes > 0
    assert result.input_bytes > 0


async def test_compress_auto_output_path(sample_jpg: Path) -> None:
    result = await compress(CompressParams(input=sample_jpg, format=ImageFormat.webp))
    expected = sample_jpg.with_suffix(".webp")
    assert result.output == expected
    assert expected.exists()
    expected.unlink()


async def test_compress_quality_int(sample_jpg: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jpg"
    result = await compress(CompressParams(input=sample_jpg, output=out, format=ImageFormat.jpeg, quality=60))
    # low quality should produce smaller file
    result_high = await compress(CompressParams(input=sample_jpg, output=tmp_path / "high.jpg", format=ImageFormat.jpeg, quality=95))
    assert result.output_bytes < result_high.output_bytes


async def test_compress_max_width_clamps(sample_jpg: Path, tmp_path: Path) -> None:
    out = tmp_path / "small.webp"
    await compress(CompressParams(input=sample_jpg, output=out, format=ImageFormat.webp, max_width=300))
    img = Image.open(out)
    assert img.width <= 300


# ─── resize ──────────────────────────────────────────────────────────────────

async def test_resize_fit(sample_jpg: Path, tmp_path: Path) -> None:
    out = tmp_path / "resized.webp"
    result = await resize(ResizeParams(input=sample_jpg, output=out, width=400, height=400, mode=ResizeMode.fit))
    img = Image.open(out)
    # fit: must be within 400×400, aspect preserved
    assert img.width <= 400
    assert img.height <= 400


async def test_resize_fill_exact(sample_jpg: Path, tmp_path: Path) -> None:
    out = tmp_path / "filled.webp"
    await resize(ResizeParams(input=sample_jpg, output=out, width=400, height=400, mode=ResizeMode.fill))
    img = Image.open(out)
    assert img.width == 400
    assert img.height == 400


async def test_resize_pad_exact(sample_jpg: Path, tmp_path: Path) -> None:
    out = tmp_path / "padded.webp"
    await resize(ResizeParams(input=sample_jpg, output=out, width=500, height=500, mode=ResizeMode.pad))
    img = Image.open(out)
    assert img.width == 500
    assert img.height == 500


# ─── convert ─────────────────────────────────────────────────────────────────

async def test_convert_png_to_webp(sample_png: Path, tmp_path: Path) -> None:
    out = tmp_path / "converted.webp"
    result = await convert(ConvertParams(input=sample_png, output=out, format=ImageFormat.webp))
    assert out.exists()
    assert result.output_bytes > 0


# ─── variants ────────────────────────────────────────────────────────────────

async def test_variants_skips_upscale(sample_jpg: Path, tmp_path: Path) -> None:
    # sample is 1200 wide; 1536 should be skipped
    result = await variants(VariantsParams(
        input=sample_jpg,
        output_dir=tmp_path,
        widths=[640, 1024, 1536],
        formats=[ImageFormat.webp],
    ))
    widths_generated = [v.width for v in result.variants]
    assert 1536 not in widths_generated  # larger than source → skipped
    assert all(w <= 1200 for w in widths_generated)


async def test_variants_multi_format(sample_jpg: Path, tmp_path: Path) -> None:
    result = await variants(VariantsParams(
        input=sample_jpg,
        output_dir=tmp_path,
        widths=[640],
        formats=[ImageFormat.webp, ImageFormat.jpeg],
    ))
    assert len(result.variants) == 2
    fmts = {v.format for v in result.variants}
    assert ImageFormat.webp in fmts
    assert ImageFormat.jpeg in fmts


# ─── lqip ────────────────────────────────────────────────────────────────────

async def test_lqip_returns_data_url(sample_jpg: Path) -> None:
    result = await lqip(LqipParams(input=sample_jpg))
    assert result.data_url.startswith("data:image/webp;base64,")
    assert result.width == 16 or result.height == 16  # longest edge = 16
    assert len(result.data_url) > 50


async def test_lqip_custom_size(sample_jpg: Path) -> None:
    result = await lqip(LqipParams(input=sample_jpg, size=32))
    assert result.width <= 32 and result.height <= 32
    assert max(result.width, result.height) == 32
