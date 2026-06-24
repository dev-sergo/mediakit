"""Programmatic builder for the Flux 2 txt2img ComfyUI workflow.

Proper Flux sampling architecture (same pattern as LTX-Video which works):
  UNETLoader → model
  CLIPLoader (mistral_3_small_flux2, type=flux2) → clip
  VAELoader → vae
  CLIPTextEncode → positive conditioning
  FluxGuidance → guidance-wrapped conditioning
  ModelSamplingFlux → timestep-patched model (critical for Flux — was causing OOM with KSampler)
  BasicScheduler → sigmas
  KSamplerSelect (euler) → sampler
  RandomNoise → noise
  CFGGuider (cfg=1.0) → guider
  SamplerCustomAdvanced → denoised latent
  VAEDecode → image
  SaveImage

Why SamplerCustomAdvanced instead of KSampler:
  KSampler applies generic SDXL-style CFG conditioning which creates
  huge intermediate tensors for Flux 2 (15360×6144 conditioning) → OOM.
  SamplerCustomAdvanced with ModelSamplingFlux applies Flux-specific
  timestep shifting which is memory-efficient even with lowvram.

Flux parameter ranges:
  guidance: 2.0–5.0  (FluxGuidance, replaces CFG)
  steps:    20–30    (distilled model)
  sampler:  euler    (not dpmpp)
  scheduler: simple  (not karras)
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class Txt2ImgFluxWorkflowParams:
    positive_prompt: str
    negative_prompt: str = ""
    unet_name: str = "flux2_dev_fp8mixed.safetensors"
    clip_name: str = "mistral_3_small_flux2_fp4_mixed.safetensors"
    vae_name: str = "flux2-vae.safetensors"
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance: float = 3.5
    max_shift: float = 1.15  # Flux timestep shift — 1.15 for ≥1MP
    base_shift: float = 0.5
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    batch_size: int = 1
    output_prefix: str = "flux_txt2img"


def build_txt2img_flux_workflow(params: Txt2ImgFluxWorkflowParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    # 1. Load Flux 2 UNet
    nodes["1"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": params.unet_name, "weight_dtype": "default"},
        "_meta": {"title": "Flux UNet"},
    }

    # 2. Load Mistral text encoder (type=flux2 confirmed valid in ComfyUI 0.18)
    nodes["2"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": params.clip_name, "type": "flux2"},
        "_meta": {"title": "Mistral CLIP"},
    }

    # 3. Load Flux VAE
    nodes["3"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": params.vae_name},
        "_meta": {"title": "Flux VAE"},
    }

    # 4. Encode positive prompt
    nodes["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["2", 0], "text": params.positive_prompt},
        "_meta": {"title": "PROMPT_POSITIVE"},
    }

    # 5. Apply Flux guidance (wraps conditioning with guidance scale)
    nodes["5"] = {
        "class_type": "FluxGuidance",
        "inputs": {"conditioning": ["4", 0], "guidance": params.guidance},
        "_meta": {"title": "Flux Guidance"},
    }

    # 6. Empty latent
    nodes["6"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": params.width, "height": params.height, "batch_size": params.batch_size},
    }

    # 7. Patch model with Flux-specific timestep shifting
    # width/height NOT passed — may be invalid in ComfyUI 0.18
    nodes["7"] = {
        "class_type": "ModelSamplingFlux",
        "inputs": {
            "model": ["1", 0],
            "max_shift": params.max_shift,
            "base_shift": params.base_shift,
            "width": params.width,
            "height": params.height,
        },
        "_meta": {"title": "Model Sampling Flux"},
    }

    # 8. Compute sigma schedule
    nodes["8"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": ["7", 0],
            "scheduler": "simple",
            "steps": params.steps,
            "denoise": 1.0,
        },
        "_meta": {"title": "Basic Scheduler"},
    }

    # 9. Select sampler
    nodes["9"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    }

    # 10. Generate noise
    nodes["10"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": params.seed},
        "_meta": {"title": "SEED"},
    }

    # 11. BasicGuider — ONE forward pass vs CFGGuider's TWO (critical for 12B Flux VRAM)
    # CFGGuider runs positive+negative simultaneously = 2x memory on 12B model = OOM
    # BasicGuider runs only the FluxGuidance-conditioned pass = 1x memory
    nodes["11"] = {
        "class_type": "BasicGuider",
        "inputs": {
            "model": ["7", 0],
            "conditioning": ["5", 0],  # FluxGuidance output
        },
    }

    # 12. Run denoising (memory-efficient path for Flux)
    nodes["12"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["10", 0],
            "guider": ["11", 0],
            "sampler": ["9", 0],
            "sigmas": ["8", 0],
            "latent_image": ["6", 0],
        },
    }

    # 13. Decode
    nodes["13"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["12", 0], "vae": ["3", 0]},
    }

    # 14. Save
    nodes["14"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["13", 0], "filename_prefix": params.output_prefix},
        "_meta": {"title": "Save"},
    }

    return nodes
