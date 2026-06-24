from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ImageFormat(StrEnum):
    jpeg = "jpeg"
    webp = "webp"
    avif = "avif"
    png = "png"

    @property
    def extension(self) -> str:
        return "jpg" if self == ImageFormat.jpeg else self.value


class Quality(StrEnum):
    low = "low"  # 60
    medium = "medium"  # 75
    high = "high"  # 85
    max = "max"  # 95

    def as_int(self) -> int:
        return {"low": 60, "medium": 75, "high": 85, "max": 95}[self.value]


def parse_quality(value: str) -> "Quality | int":
    """Parse 'high'/'low'/... or a numeric string like '82' into Quality | int."""
    try:
        return Quality(value)
    except ValueError:
        return int(value)


class ResizeMode(StrEnum):
    fit = "fit"  # contain within bounds, no crop, preserve aspect
    fill = "fill"  # cover: fill bounds, center-crop excess
    smart_crop = "smart_crop"  # saliency-aware crop
    pad = "pad"  # fit + letterbox padding, no crop


# ─── Compress ────────────────────────────────────────────────────────────────


class CompressParams(BaseModel):
    input: Path
    output: Path | None = None  # derived from input + format if None
    format: ImageFormat = ImageFormat.webp
    quality: Quality | int = Quality.high
    max_width: int | None = None  # never upscale; clamp width if needed
    strip_metadata: bool = True

    def quality_int(self) -> int:
        if isinstance(self.quality, Quality):
            return self.quality.as_int()
        return self.quality


class CompressResult(BaseModel):
    output: Path
    input_bytes: int
    output_bytes: int
    savings_pct: float


# ─── Resize ──────────────────────────────────────────────────────────────────


class ResizeParams(BaseModel):
    input: Path
    output: Path | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: ResizeMode = ResizeMode.fit
    pad_color: tuple[int, int, int] = (255, 255, 255)
    format: ImageFormat | None = None  # if None, keep source format


class ResizeResult(BaseModel):
    output: Path
    width: int
    height: int


# ─── Convert ─────────────────────────────────────────────────────────────────


class ConvertParams(BaseModel):
    input: Path
    output: Path | None = None
    format: ImageFormat
    quality: Quality | int = Quality.high

    def quality_int(self) -> int:
        if isinstance(self.quality, Quality):
            return self.quality.as_int()
        return self.quality


class ConvertResult(BaseModel):
    output: Path
    input_bytes: int
    output_bytes: int


# ─── Variants ────────────────────────────────────────────────────────────────


class VariantsParams(BaseModel):
    input: Path
    output_dir: Path | None = None
    widths: list[int] = Field(default=[640, 768, 1024, 1280, 1536], min_length=1)
    formats: list[ImageFormat] = Field(default=[ImageFormat.webp])
    quality: Quality | int = Quality.high
    stem: str | None = None  # output filename stem; defaults to input stem

    def quality_int(self) -> int:
        if isinstance(self.quality, Quality):
            return self.quality.as_int()
        return self.quality

    @model_validator(mode="after")
    def widths_sorted(self) -> "VariantsParams":
        self.widths = sorted(set(self.widths))
        return self


class VariantFile(BaseModel):
    path: Path
    width: int
    height: int
    format: ImageFormat


class VariantsResult(BaseModel):
    variants: list[VariantFile]


# ─── LQIP ────────────────────────────────────────────────────────────────────


class LqipParams(BaseModel):
    input: Path
    size: int = Field(default=16, gt=0, le=64)  # longest edge in pixels


class LqipResult(BaseModel):
    data_url: str  # data:image/webp;base64,...
    width: int
    height: int
