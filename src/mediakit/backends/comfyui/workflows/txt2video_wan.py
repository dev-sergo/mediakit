"""Wan 2.1 txt2video workflow builder.

Based on wan_phantom_supercar.json (simplified — no reactor face swap, no subject image).
Architecture:
  UNETLoader (wan2.1 unet) → model
  CLIPLoader (umt5, type=wan) → clip
  VAELoader (wan vae) → vae
  CLIPTextEncode ×2 → conditioning
  EmptyHuanyuanVideoLatentVideo (or EmptyWanLatentVideo) → latent
  KSampler (uni_pc, simple) → denoised latent
  VAEDecodeTiled → image frames
  CreateVideo → video bytes
  SaveVideo → output file

VERIFY on GPU box:
  - UNETLoader unet path: split_files/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors
  - CLIPLoader type: "wan" (verify this matches installed node version)
  - VAELoader vae path: split_files/vae/wan_2.1_vae.safetensors
  - EmptyLatentImage node name for Wan: check if it's EmptyWanLatentVideo or WanEmptyLatent
    — the JSON used WanPhantomSubjectToVideo (different from txt2video); for pure txt2video
    the latent node might differ. Fall back to EmptyLatentImage if unsure.
  - KSampler sampler: "uni_pc" with "simple" scheduler works for Wan.

Length note: Wan uses multiples of 4+1 (5, 9, 17, 25, 33, 49, 65, 81...).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class WanTxt2VideoParams:
    positive_prompt: str
    negative_prompt: str = (
        "Overexposure, static, blurred details, subtitles, ugly, worst quality, watermark"
    )
    unet_name: str = "split_files/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors"
    clip_name: str = "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_name: str = "split_files/vae/wan_2.1_vae.safetensors"
    width: int = 832
    height: int = 480
    length: int = 49  # frames — multiples of 4+1
    fps: float = 16.0
    steps: int = 20
    cfg: float = 6.5
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    output_prefix: str = "video/wan"


def build_wan_txt2video_workflow(params: WanTxt2VideoParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": params.unet_name, "weight_dtype": "default"},
        "_meta": {"title": "Wan UNet Loader"},
    }
    nodes["2"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": params.clip_name, "type": "wan"},
        "_meta": {"title": "UMT5 CLIP Loader"},
    }
    nodes["3"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": params.vae_name},
        "_meta": {"title": "Wan VAE Loader"},
    }
    nodes["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["2", 0], "text": params.positive_prompt},
        "_meta": {"title": "PROMPT_POSITIVE"},
    }
    nodes["5"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["2", 0], "text": params.negative_prompt},
        "_meta": {"title": "PROMPT_NEGATIVE"},
    }
    # EmptyHunyuanLatentVideo is structurally compatible with Wan (same 16ch 4x-temporal 3D latent).
    nodes["6"] = {
        "class_type": "EmptyHunyuanLatentVideo",
        "inputs": {
            "width": params.width,
            "height": params.height,
            "length": params.length,
            "batch_size": 1,
        },
        "_meta": {"title": "Empty Latent"},
    }
    nodes["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": params.seed,
            "steps": params.steps,
            "cfg": params.cfg,
            "sampler_name": "uni_pc",
            "scheduler": "simple",
            "denoise": 1.0,
        },
        "_meta": {"title": "SEED"},
    }
    nodes["8"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["3", 0],
            "tile_size": 256,
            "overlap": 64,
            "temporal_size": 32,
            "temporal_overlap": 4,
        },
    }
    nodes["9"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["8", 0], "fps": params.fps},
    }
    nodes["10"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["9", 0],
            "filename_prefix": params.output_prefix,
            "format": "auto",
            "codec": "auto",
        },
        "_meta": {"title": "Save"},
    }

    return nodes
