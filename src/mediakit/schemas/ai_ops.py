from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ─── Background removal ───────────────────────────────────────────────────────


class BiRefNetModel(StrEnum):
    hr = "BiRefNet-HR"
    hr_matting = "BiRefNet-HR-matting"
    general = "BiRefNet-general"
    dynamic = "BiRefNet_dynamic"
    lite_2k = "BiRefNet_lite-2K"
    matting = "BiRefNet-matting"
    portrait = "BiRefNet-portrait"


class BgRemoveParams(BaseModel):
    input: Path
    output: Path | None = None
    model: BiRefNetModel = BiRefNetModel.hr
    background_mode: Literal["transparent", "color"] = "transparent"
    background_color: str = "#FFFFFF"
    mask_blur: int = Field(default=0, ge=0, le=64)
    mask_offset: int = Field(default=0, ge=-20, le=20)
    refine_foreground: bool = True
    width: int = Field(default=0, ge=0, le=4096)  # 0 = preserve source
    height: int = Field(default=0, ge=0, le=4096)


class BgRemoveResult(BaseModel):
    output: Path


# ─── Upscale ──────────────────────────────────────────────────────────────────


class UpscaleModel(StrEnum):
    nmkd = "4x_NMKD-Siax_200k.pth"
    realesrgan = "RealESRGAN_x4.pth"


class UpscaleParams(BaseModel):
    input: Path
    output: Path | None = None
    model: UpscaleModel = UpscaleModel.nmkd
    scale: float = Field(default=2.0, ge=1.5, le=4.0)


class UpscaleResult(BaseModel):
    output: Path


# ─── txt2img ─────────────────────────────────────────────────────────────────


class Txt2ImgParams(BaseModel):
    prompt: str
    negative_prompt: str = ""
    # backend selects the generation architecture:
    #   "sdxl" — CheckpointLoaderSimple (RealVisXL etc.), cfg 5–9, dpmpp_2m+karras
    #   "flux"  — UNETLoader + DualCLIPLoader + VAELoader, guidance 2–5, euler+simple
    backend: Literal["sdxl", "flux"] = "sdxl"
    checkpoint: str = "RealVisXL_V5.0_inpainting.safetensors"
    width: int = Field(default=1024, ge=512, le=2048, multiple_of=8)
    height: int = Field(default=1024, ge=512, le=2048, multiple_of=8)
    steps: int = Field(default=25, ge=1, le=100)
    cfg: float = Field(default=7.5, ge=1.0, le=20.0)  # used as guidance for Flux
    seed: int = -1  # -1 = random; resolved to a concrete value inside the op
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    output: Path | None = None


class Txt2ImgResult(BaseModel):
    output: Path
    seed: int  # actual seed used (useful when seed was random)


# ─── img_edit ────────────────────────────────────────────────────────────────


class ImgEditParams(BaseModel):
    input: Path
    prompt: str
    negative_prompt: str = ""
    # backend:
    #   "sdxl"  — SDXL inpainting (RealVisXL), cfg 5-9, 20+ steps
    #   "qwen"  — Qwen Image Edit 2511, instruction-based, cfg~1.0, 4 steps (Lightning)
    backend: Literal["sdxl", "qwen"] = "sdxl"
    checkpoint: str = "RealVisXL_V5.0_inpainting.safetensors"
    width: int = Field(default=1024, ge=512, le=2048, multiple_of=8)
    height: int = Field(default=1024, ge=512, le=2048, multiple_of=8)
    steps: int = Field(default=25, ge=1, le=100)
    cfg: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int = -1  # -1 = random; resolved to a concrete value inside the op
    mask_target: Literal["background", "full"] = "background"
    mask_blur: int = Field(default=4, ge=0, le=64)
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)  # Qwen Lightning LoRA weight
    output: Path | None = None


class ImgEditResult(BaseModel):
    output: Path
    seed: int
