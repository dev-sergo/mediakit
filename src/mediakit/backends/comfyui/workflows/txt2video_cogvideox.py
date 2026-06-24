"""CogVideoX txt2video workflow builder (kijai ComfyUI-CogVideoXWrapper).

CogVideoX-5B is trained at 8 fps / 49 frames = ~6 seconds in a single window —
roughly 3x LTX-Video's seamless window. For clips up to the native window no
stitching is needed; see pipelines/seamless_video.py for longer clips.

Architecture (kijai wrapper, verified against cogvideox_1_0_5b_T2V_02.json):
  DownloadAndLoadCogVideoModel → model (output 0), vae (output 1)
  CLIPLoader (t5xxl, type=sd3) → clip
  CogVideoTextEncode ×2 → positive / negative conditioning
  EmptyLatentImage (width, height) → samples latent (provides H/W shape to sampler)
  CogVideoSampler (model, positive, negative, samples, num_frames, ...) → latent
  CogVideoDecode (vae, samples) → image frames
  CreateVideo → video bytes
  SaveVideo → output file

Notes:
  - CogVideoSampler does NOT take width/height directly — it reads H/W from the
    `samples` latent shape. EmptyLatentImage provides this shape.
  - num_frames is a required direct input on CogVideoSampler.
  - CogVideoDecode takes `vae` (output 1 of model loader) and `samples` only —
    no `pipeline` input.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class CogVideoXTxt2VideoParams:
    positive_prompt: str
    negative_prompt: str = "worst quality, blurry, jittery, distorted, watermark, text, low quality"
    model_id: str = "THUDM/CogVideoX-5b"
    clip_name: str = "t5xxl_fp8_e4m3fn.safetensors"
    precision: str = "bf16"
    width: int = 720
    height: int = 480
    length: int = 49  # frames — must be 8n+1 for CogVideoX-5B
    fps: float = 8.0  # CogVideoX-5B is trained at 8 fps
    steps: int = 50
    cfg: float = 6.0
    scheduler: str = "CogVideoXDDIM"
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    output_prefix: str = "video/cogvideox"


def build_cogvideox_txt2video_workflow(params: CogVideoXTxt2VideoParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "DownloadAndLoadCogVideoModel",
        "inputs": {
            "model": params.model_id,
            "precision": params.precision,
            "fp8_transformer": "disabled",
            "enable_sequential_cpu_offload": False,
        },
        "_meta": {"title": "CogVideoX Model"},
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
    # EmptyLatentImage provides the H/W shape that CogVideoSampler reads internally.
    nodes["5"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": params.width,
            "height": params.height,
            "batch_size": 1,
        },
        "_meta": {"title": "Empty Latent"},
    }
    nodes["6"] = {
        "class_type": "CogVideoSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["3", 0],
            "negative": ["4", 0],
            "samples": ["5", 0],
            "num_frames": params.length,
            "steps": params.steps,
            "cfg": params.cfg,
            "seed": params.seed,
            "scheduler": params.scheduler,
            "denoise_strength": 1.0,
        },
        "_meta": {"title": "SEED"},
    }
    nodes["7"] = {
        "class_type": "CogVideoDecode",
        "inputs": {
            "vae": ["1", 1],
            "samples": ["6", 0],
            "enable_vae_tiling": True,
            "auto_tile_size": True,
            "tile_sample_min_height": 240,
            "tile_sample_min_width": 360,
            "tile_overlap_factor_height": 0.2,
            "tile_overlap_factor_width": 0.2,
        },
        "_meta": {"title": "Decode"},
    }
    nodes["8"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["7", 0], "fps": params.fps},
    }
    nodes["9"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["8", 0],
            "filename_prefix": params.output_prefix,
            "format": "auto",
            "codec": "auto",
        },
        "_meta": {"title": "Save"},
    }

    return nodes
