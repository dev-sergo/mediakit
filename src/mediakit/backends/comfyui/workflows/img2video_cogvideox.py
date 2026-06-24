"""CogVideoX img2video workflow builder (kijai ComfyUI-CogVideoXWrapper).

Uses the CogVideoX-5B-I2V checkpoint: the input image is VAE-encoded into the
first-frame conditioning latent, so the generated clip starts exactly on the
supplied frame. This is what makes seamless continuation possible — feed the
last frame of segment N as the input image of segment N+1 (see
pipelines/seamless_video.py).

Architecture (kijai wrapper):
  DownloadAndLoadCogVideoModel (THUDM/CogVideoX-5b-I2V) → model (output 0), vae (output 1)
  CLIPLoader (t5xxl, type=sd3) → clip
  CogVideoTextEncode ×2 → positive / negative conditioning
  LoadImage → image
  CogVideoImageEncode (vae, image) → image_cond_latents
  EmptyLatentImage (width, height) → samples (provides H/W shape to CogVideoSampler)
  CogVideoSampler (model, positive, negative, samples, image_cond_latents, num_frames, ...)
  CogVideoDecode (vae, samples) → image frames
  CreateVideo → SaveVideo

Notes:
  - CogVideoSampler reads H/W from the `samples` latent shape, not direct inputs.
    EmptyLatentImage carries the spatial dimensions the sampler needs.
  - CogVideoImageEncode takes `vae` (output 1) and `image` — no `pipeline` input.
  - CogVideoDecode takes `vae` and `samples` only — no `pipeline` input.
  - model_id must be the I2V checkpoint ("THUDM/CogVideoX-5b-I2V").
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class CogVideoXImg2VideoParams:
    positive_prompt: str
    image_filename: str              # filename on ComfyUI server after upload
    negative_prompt: str = (
        "worst quality, blurry, jittery, distorted, morphing, deformed, watermark"
    )
    model_id: str = "THUDM/CogVideoX-5b-I2V"
    clip_name: str = "t5xxl_fp8_e4m3fn.safetensors"
    precision: str = "bf16"
    width: int = 720
    height: int = 480
    length: int = 49       # frames — must be 8n+1 for CogVideoX-5B
    fps: float = 8.0
    steps: int = 50
    cfg: float = 6.0
    scheduler: str = "CogVideoXDDIM"
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    output_prefix: str = "video/cogvideox_i2v"


def build_cogvideox_img2video_workflow(params: CogVideoXImg2VideoParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "DownloadAndLoadCogVideoModel",
        "inputs": {
            "model": params.model_id,
            "precision": params.precision,
            "fp8_transformer": "disabled",
            "enable_sequential_cpu_offload": False,
        },
        "_meta": {"title": "CogVideoX I2V Model"},
    }
    nodes["2"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": params.clip_name, "type": "sd3"},
        "_meta": {"title": "T5 CLIP Loader"},
    }
    nodes["3"] = {
        "class_type": "CogVideoTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "prompt": params.positive_prompt,
            "strength": 1.0,
            "force_offload": True,
        },
        "_meta": {"title": "PROMPT_POSITIVE"},
    }
    nodes["4"] = {
        "class_type": "CogVideoTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "prompt": params.negative_prompt,
            "strength": 1.0,
            "force_offload": True,
        },
        "_meta": {"title": "PROMPT_NEGATIVE"},
    }
    nodes["5"] = {
        "class_type": "LoadImage",
        "inputs": {"image": params.image_filename},
        "_meta": {"title": "Input Image"},
    }
    # CogVideoImageEncode: vae (output 1 of model loader) + image → image_cond_latents
    nodes["6"] = {
        "class_type": "CogVideoImageEncode",
        "inputs": {
            "vae": ["1", 1],
            "image": ["5", 0],
        },
        "_meta": {"title": "Image Encode"},
    }
    # EmptyLatentImage carries the H/W shape that CogVideoSampler reads internally.
    nodes["7"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": params.width,
            "height": params.height,
            "batch_size": 1,
        },
        "_meta": {"title": "Empty Latent"},
    }
    nodes["8"] = {
        "class_type": "CogVideoSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["3", 0],
            "negative": ["4", 0],
            "samples": ["7", 0],
            "image_cond_latents": ["6", 0],
            "num_frames": params.length,
            "steps": params.steps,
            "cfg": params.cfg,
            "seed": params.seed,
            "scheduler": params.scheduler,
            "denoise_strength": 1.0,
        },
        "_meta": {"title": "SEED"},
    }
    nodes["9"] = {
        "class_type": "CogVideoDecode",
        "inputs": {
            "vae": ["1", 1],
            "samples": ["8", 0],
            "enable_vae_tiling": True,
            "auto_tile_size": True,
            "tile_sample_min_height": 240,
            "tile_sample_min_width": 360,
            "tile_overlap_factor_height": 0.2,
            "tile_overlap_factor_width": 0.2,
        },
        "_meta": {"title": "Decode"},
    }
    nodes["10"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["9", 0], "fps": params.fps},
    }
    nodes["11"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["10", 0],
            "filename_prefix": params.output_prefix,
            "format": "auto",
            "codec": "auto",
        },
        "_meta": {"title": "Save"},
    }

    return nodes
