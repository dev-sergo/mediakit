"""Write generation metadata into image files.

Strategy:
  PNG  → PngInfo text chunks (lossless, visible in Preview/exiftool)
  JPEG → EXIF via piexif: ImageDescription (prompt) + UserComment (params JSON)
         Both fields are visible in macOS Finder → Get Info → More Info.
  WebP → JSON sidecar only (Pillow XMP support is limited)
  All  → .json sidecar file next to the image (always reliable)

Usage:
    write_metadata(path, seed=42, steps=25, cfg=7.5, checkpoint="...", prompt="...")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from PIL import Image, PngImagePlugin

log = structlog.get_logger(__name__)

_PNG_PREFIX = "mediakit:"
_EXIF_IMAGE_DESCRIPTION = 0x010E  # visible in Finder "Description"
_EXIF_USER_COMMENT = 0x9286  # visible in Finder "More Info"
_EXIF_DATE_TIME = 0x0132


def write_metadata(path: Path, **params: Any) -> None:
    """Embed generation params in image file + write .json sidecar.

    Never raises — metadata failure must not break a generation.
    """
    meta = {k: v for k, v in params.items() if v is not None}
    meta["generated_at"] = datetime.now().isoformat(timespec="seconds")

    try:
        _embed(path, meta)
    except Exception as exc:
        log.warning("metadata.embed_failed", path=str(path), error=str(exc))
    try:
        _sidecar(path, meta)
    except Exception as exc:
        log.warning("metadata.sidecar_failed", path=str(path), error=str(exc))


def _embed(path: Path, meta: dict[str, Any]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".png":
        _embed_png(path, meta)
    elif suffix in (".jpg", ".jpeg"):
        _embed_jpeg(path, meta)


def _embed_png(path: Path, meta: dict[str, Any]) -> None:
    img = Image.open(path)
    info = PngImagePlugin.PngInfo()
    for key, val in meta.items():
        info.add_text(f"{_PNG_PREFIX}{key}", str(val))
    img.save(path, pnginfo=info)


def _embed_jpeg(path: Path, meta: dict[str, Any]) -> None:
    try:
        import piexif
    except ImportError:
        return  # piexif not installed — fall back to sidecar only

    try:
        exif_dict = piexif.load(str(path))
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    # ImageDescription (0th IFD) — visible in Finder
    prompt = str(meta.get("prompt", ""))[:200]
    if prompt:
        exif_dict["0th"][_EXIF_IMAGE_DESCRIPTION] = prompt.encode("utf-8")

    # DateTime (0th IFD)
    ts = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    exif_dict["0th"][_EXIF_DATE_TIME] = ts.encode("ascii")

    # UserComment (Exif IFD) — stores params as compact JSON, visible in Finder More Info
    summary = {k: v for k, v in meta.items() if k != "prompt"}
    comment_text = json.dumps(summary, ensure_ascii=False)
    # piexif UserComment format: 8-byte charset header + text
    exif_dict["Exif"][_EXIF_USER_COMMENT] = b"UNICODE\x00" + comment_text.encode("utf-16-le")

    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, str(path))  # in-place, no recompression


def _sidecar(path: Path, meta: dict[str, Any]) -> None:
    sidecar = path.with_name(path.name + ".json")
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
