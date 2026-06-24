"""Programmatic builder for the txt2img ComfyUI workflow (SDXL).

CheckpointLoaderSimple → CLIPTextEncode ×2 → EmptyLatentImage → KSampler → VAEDecode → SaveImage
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class Txt2ImgWorkflowParams:
    positive_prompt: str
    negative_prompt: str = ""
    checkpoint: str = "RealVisXL_V5.0_inpainting.safetensors"
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg: float = 7.5
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    sampler_name: str = "dpmpp_2m"
    scheduler: str = "karras"
    batch_size: int = 1
    output_prefix: str = "txt2img"


def build_txt2img_workflow(params: Txt2ImgWorkflowParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": params.checkpoint},
        "_meta": {"title": "Checkpoint"},
    }
    nodes["2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": params.positive_prompt},
        "_meta": {"title": "PROMPT_POSITIVE"},
    }
    nodes["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": params.negative_prompt},
        "_meta": {"title": "PROMPT_NEGATIVE"},
    }
    nodes["4"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": params.width, "height": params.height, "batch_size": params.batch_size},
    }
    nodes["5"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": params.seed,
            "steps": params.steps,
            "cfg": params.cfg,
            "sampler_name": params.sampler_name,
            "scheduler": params.scheduler,
            "denoise": 1.0,
        },
        "_meta": {"title": "SEED"},
    }
    nodes["6"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    }
    nodes["7"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": params.output_prefix},
        "_meta": {"title": "Save"},
    }

    return nodes
