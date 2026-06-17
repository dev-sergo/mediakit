import io
from pathlib import Path
from typing import Any

from PIL import Image

from mediakit.schemas.ops import ImageFormat

_SAVE_KWARGS: dict[ImageFormat, dict[str, Any]] = {
    ImageFormat.jpeg: {"optimize": True, "progressive": True},
    ImageFormat.webp: {"method": 6},
    ImageFormat.avif: {},
    ImageFormat.png: {"optimize": True},
}

_PIL_FORMAT: dict[ImageFormat, str] = {
    ImageFormat.jpeg: "JPEG",
    ImageFormat.webp: "WEBP",
    ImageFormat.avif: "AVIF",
    ImageFormat.png: "PNG",
}


def resolve_output(input_path: Path, fmt: ImageFormat, output: Path | None) -> Path:
    if output is not None:
        return output
    return input_path.with_suffix(f".{fmt.extension}")


def encode(img: Image.Image, path: Path, fmt: ImageFormat, quality: int) -> None:
    """Write `img` to `path` in the requested format."""
    pil_fmt = _PIL_FORMAT[fmt]
    kwargs: dict[str, Any] = dict(_SAVE_KWARGS.get(fmt, {}))

    if fmt != ImageFormat.png:
        kwargs["quality"] = quality

    # PNG and AVIF don't use progressive; JPEG needs RGB
    if fmt == ImageFormat.jpeg and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif fmt == ImageFormat.webp and img.mode == "RGBA":
        pass  # WEBP supports RGBA natively
    elif fmt in (ImageFormat.jpeg, ImageFormat.webp) and img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=pil_fmt, **kwargs)


def strip_metadata(img: Image.Image) -> Image.Image:
    """Return a copy of `img` without EXIF/ICC metadata.

    Pillow does not embed EXIF on save unless explicitly passed — copying
    the pixel data ensures no accidental metadata leaks through .info dict.
    """
    clean = Image.new(img.mode, img.size)
    clean.paste(img)
    return clean


def encode_bytes(img: Image.Image, fmt: ImageFormat, quality: int) -> bytes:
    buf = io.BytesIO()
    kwargs: dict[str, Any] = dict(_SAVE_KWARGS.get(fmt, {}))
    if fmt != ImageFormat.png:
        kwargs["quality"] = quality
    img.save(buf, format=_PIL_FORMAT[fmt], **kwargs)
    return buf.getvalue()
