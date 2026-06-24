"""LTX-Video txt2video workflow builder.

Based on ltxv_t2v_quality.json (image-service/workflows/).
Architecture:
  CheckpointLoaderSimple (LTXV checkpoint) → model + vae
  CLIPLoader (t5xxl, type=ltxv) → clip
  CLIPTextEncode ×2 → conditioning
  LTXVConditioning (frame_rate) → timed conditioning
  EmptyLTXVLatentVideo → latent
  ModelSamplingLTXV → patched model
  LTXVScheduler + KSamplerSelect + RandomNoise + CFGGuider → sampler setup
  SamplerCustomAdvanced → denoised latent
  VAEDecodeTiled → image frames
  CreateVideo → video bytes
  SaveVideo → output file

VERIFY on GPU box:
  - CheckpointLoaderSimple path: LTXV/ltxv-13b-0.9.8-distilled-fp8.safetensors
  - CLIPLoader type: "ltxv" (must match installed node version)
  - LTXVConditioning, EmptyLTXVLatentVideo, LTXVScheduler, ModelSamplingLTXV
    are from the LTXV custom node — ensure it is installed and up to date.
  - CreateVideo, SaveVideo output format: check codec availability.

Length note (LTX constraint): valid values are 9 + 8*n (9, 17, 25, 33, 49, 65, 97, 161...).
Frame rate note: 24 fps for cinematic, 25 fps to match ltxv default.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class LtxvTxt2VideoParams:
    positive_prompt: str
    negative_prompt: str = "worst quality, blurry, jittery, distorted, watermark, text"
    checkpoint: str = "LTXV/ltxv-13b-0.9.8-distilled-fp8.safetensors"
    clip_name: str = "t5xxl_fp8_e4m3fn.safetensors"
    width: int = 768
    height: int = 512
    length: int = 49  # frames — must be 9+8n
    fps: float = 24.0
    steps: int = 15
    cfg: float = 2.5
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    max_shift: float = 2.05
    base_shift: float = 0.95
    output_prefix: str = "video/ltxv"


def build_ltxv_txt2video_workflow(params: LtxvTxt2VideoParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": params.checkpoint},
        "_meta": {"title": "LTX Checkpoint"},
    }
    nodes["2"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": params.clip_name, "type": "ltxv"},
        "_meta": {"title": "T5 CLIP Loader"},
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
        "_meta": {"title": "LTXV Conditioning"},
    }
    nodes["6"] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": params.width,
            "height": params.height,
            "length": params.length,
            "batch_size": 1,
        },
    }
    nodes["7"] = {
        "class_type": "ModelSamplingLTXV",
        "inputs": {
            "model": ["1", 0],
            "max_shift": params.max_shift,
            "base_shift": params.base_shift,
        },
    }
    nodes["8"] = {
        "class_type": "LTXVScheduler",
        "inputs": {
            "steps": params.steps,
            "max_shift": params.max_shift,
            "base_shift": params.base_shift,
            "stretch": True,
            "terminal": 0.1,
            "latent": ["6", 0],
        },
    }
    nodes["9"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    }
    nodes["10"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": params.seed},
        "_meta": {"title": "SEED"},
    }
    nodes["11"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["7", 0],
            "positive": ["5", 0],
            "negative": ["5", 1],
            "cfg": params.cfg,
        },
    }
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
    nodes["13"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["12", 0],
            "vae": ["1", 2],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 8,
        },
    }
    nodes["14"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["13", 0],
            "fps": params.fps,
        },
    }
    nodes["15"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["14", 0],
            "filename_prefix": params.output_prefix,
            "format": "auto",
            "codec": "auto",
        },
        "_meta": {"title": "Save"},
    }

    return nodes
