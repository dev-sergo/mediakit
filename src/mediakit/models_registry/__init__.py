"""Non-blocking model presence check — warns on startup if expected models are absent."""
from __future__ import annotations

from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

# Maps ComfyUI model sub-directory → list of expected filenames.
# Update when new ops/checkpoints are added.
REQUIRED_MODELS: dict[str, list[str]] = {
    # SDXL / SD 1.5 checkpoints
    "checkpoints": [
        "RealVisXL_V5.0_inpainting.safetensors",
    ],
    # Upscale
    "upscale_models": [
        "4x_NMKD-Siax_200k.pth",
        "RealESRGAN_x4.pth",
    ],
    # Flux 2 (new ComfyUI API format)
    "diffusion_models": [
        "flux2_dev_fp8mixed.safetensors",
    ],
    # Qwen Image Edit
    # diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors is in the same dir
    # text_encoders are separate
    "text_encoders": [
        "t5xxl_fp8_e4m3fn.safetensors",       # Flux + LTX-Video
        "qwen_2.5_vl_7b_fp8_scaled.safetensors",  # Qwen Image Edit
    ],
    # VAE
    "vae": [
        "flux2-vae.safetensors",
        "qwen_image_vae.safetensors",
    ],
    # LoRA
    "loras": [
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    ],
}


def check_models(models_dir: Path) -> dict[str, list[str]]:
    """Return {category: [missing_filename, ...]} for models not found on disk."""
    missing: dict[str, list[str]] = {}
    for category, names in REQUIRED_MODELS.items():
        absent = [n for n in names if not (models_dir / category / n).exists()]
        if absent:
            missing[category] = absent
    return missing


def warn_missing_models(models_dir: Path) -> None:
    """Log a warning for each model file that is not found. Does not raise."""
    missing = check_models(models_dir)
    for category, names in missing.items():
        for name in names:
            log.warning(
                "mediakit.model_missing",
                category=category,
                model=name,
                expected_path=str(models_dir / category / name),
            )
    if not missing:
        log.info("mediakit.models_ok", models_dir=str(models_dir))
