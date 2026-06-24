"""LTX-Video img2video workflow builder.

Based on ltxv_i2v_portrait.json (image-service/workflows/).
Key difference from txt2video: uses LTXVImgToVideo instead of EmptyLTXVLatentVideo.
This animates an existing image rather than generating from scratch.

Use cases: portrait animation, product animation, travel photo animation.

VERIFY on GPU box:
  - LTXVImgToVideo node — takes image + outputs latent conditioned on input image.
  - strength: 0.9 = strong motion, 0.5 = subtle. Adjust per use case.
  - LoadImage node name: might need the image to be uploaded first via ComfyUI upload API.
    mediakit client handles this via comfy.upload_image().
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class LtxvImg2VideoParams:
    positive_prompt: str
    image_filename: str  # filename on ComfyUI server after upload
    negative_prompt: str = "worst quality, blurry, jittery, distorted, morphing, deformed"
    checkpoint: str = "LTXV/ltxv-13b-0.9.8-distilled-fp8.safetensors"
    clip_name: str = "t5xxl_fp8_e4m3fn.safetensors"
    width: int = 768
    height: int = 512
    length: int = 49
    fps: float = 24.0
    steps: int = 15
    cfg: float = 2.0
    strength: float = 0.9  # animation strength (0.5=subtle, 0.9=strong)
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    max_shift: float = 2.05
    base_shift: float = 0.95
    output_prefix: str = "video/ltxv_i2v"


def build_ltxv_img2video_workflow(params: LtxvImg2VideoParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": params.checkpoint},
        "_meta": {"title": "LTX Checkpoint"},
    }
    nodes["2"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": params.clip_name, "type": "ltxv"},
    }
    nodes["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["2", 0], "text": params.positive_prompt},
        "_meta": {"title": "PROMPT_POSITIVE"},
    }
    nodes["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["2", 0], "text": params.negative_prompt},
        "_meta": {"title": "PROMPT_NEGATIVE"},
    }
    nodes["5"] = {
        "class_type": "LTXVConditioning",
        "inputs": {
            "positive": ["3", 0],
            "negative": ["4", 0],
            "frame_rate": params.fps,
        },
    }
    nodes["6"] = {
        "class_type": "LoadImage",
        "inputs": {"image": params.image_filename},
        "_meta": {"title": "Input Image"},
    }
    nodes["7"] = {
        "class_type": "LTXVImgToVideo",
        "inputs": {
            "positive": ["5", 0],
            "negative": ["5", 1],
            "vae": ["1", 2],
            "image": ["6", 0],
            "width": params.width,
            "height": params.height,
            "length": params.length,
            "batch_size": 1,
            "strength": params.strength,
        },
    }
    nodes["8"] = {
        "class_type": "ModelSamplingLTXV",
        "inputs": {
            "model": ["1", 0],
            "max_shift": params.max_shift,
            "base_shift": params.base_shift,
        },
    }
    nodes["9"] = {
        "class_type": "LTXVScheduler",
        "inputs": {
            "steps": params.steps,
            "max_shift": params.max_shift,
            "base_shift": params.base_shift,
            "stretch": True,
            "terminal": 0.1,
            "latent": ["7", 2],
        },
    }
    nodes["10"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    nodes["11"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": params.seed},
        "_meta": {"title": "SEED"},
    }
    nodes["12"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["8", 0],
            "positive": ["7", 0],
            "negative": ["7", 1],
            "cfg": params.cfg,
        },
    }
    nodes["13"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["11", 0],
            "guider": ["12", 0],
            "sampler": ["10", 0],
            "sigmas": ["9", 0],
            "latent_image": ["7", 2],
        },
    }
    nodes["14"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["13", 0],
            "vae": ["1", 2],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 8,
        },
    }
    nodes["15"] = {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": params.fps}}
    nodes["16"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["15", 0],
            "filename_prefix": params.output_prefix,
            "format": "auto",
            "codec": "auto",
        },
        "_meta": {"title": "Save"},
    }

    return nodes
