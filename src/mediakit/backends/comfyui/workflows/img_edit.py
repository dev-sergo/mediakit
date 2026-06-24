"""Programmatic builder for the img_edit ComfyUI workflow.

SDXL inpainting + BiRefNet mask. Replaces masked area (background by default)
while keeping the subject pixel-perfect.

  LoadImage → ImageScale → BiRefNetRMBG → [InvertMask] → VAEEncodeForInpaint
  CheckpointLoaderSimple → CLIPTextEncode ×2
  KSampler → VAEDecode → SaveImage
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]

_DEFAULT_CHECKPOINT = "RealVisXL_V5.0_inpainting.safetensors"


@dataclass(frozen=True)
class ImgEditWorkflowParams:
    image_filename: str
    positive_prompt: str
    negative_prompt: str = ""
    checkpoint: str = _DEFAULT_CHECKPOINT
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg: float = 7.5
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    sampler_name: str = "dpmpp_2m"
    scheduler: str = "karras"
    denoise: float = 1.0
    # BiRefNet
    birefnet_model: str = "BiRefNet-HR"
    mask_blur: int = 4
    mask_offset: int = 2
    refine_foreground: bool = True
    # "background" = inpaint everything outside subject; "full" = inpaint whole image
    mask_target: str = "background"
    grow_mask_by: int = 6
    output_prefix: str = "img_edit"


def build_img_edit_workflow(params: ImgEditWorkflowParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "LoadImage",
        "inputs": {"image": params.image_filename},
        "_meta": {"title": "INPUT_IMAGE"},
    }
    nodes["2"] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["1", 0],
            "upscale_method": "lanczos",
            "width": params.width,
            "height": params.height,
            "crop": "disabled",
        },
    }

    # BiRefNet: outputs (image_with_bg, mask)
    nodes["3"] = {
        "class_type": "BiRefNetRMBG",
        "inputs": {
            "image": ["2", 0],
            "model": params.birefnet_model,
            "mask_blur": params.mask_blur,
            "mask_offset": params.mask_offset,
            "invert_output": False,
            "refine_foreground": params.refine_foreground,
            "background": "Alpha",
            "background_color": "#000000",
        },
        "_meta": {"title": "BiRefNet"},
    }

    # mask_terminal: subject mask (1=subject). For background inpainting we invert it.
    mask_terminal: list[Any]
    if params.mask_target == "background":
        nodes["4"] = {
            "class_type": "InvertMask",
            "inputs": {"mask": ["3", 1]},
            "_meta": {"title": "Background Mask"},
        }
        mask_terminal = ["4", 0]
    else:
        # full-image inpaint: solid white mask
        nodes["4"] = {
            "class_type": "SolidMask",
            "inputs": {"value": 1.0, "width": params.width, "height": params.height},
            "_meta": {"title": "Full Mask"},
        }
        mask_terminal = ["4", 0]

    nodes["5"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": params.checkpoint},
        "_meta": {"title": "Checkpoint"},
    }
    nodes["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["5", 1], "text": params.positive_prompt},
        "_meta": {"title": "PROMPT_POSITIVE"},
    }
    nodes["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["5", 1], "text": params.negative_prompt},
        "_meta": {"title": "PROMPT_NEGATIVE"},
    }
    nodes["8"] = {
        "class_type": "VAEEncodeForInpaint",
        "inputs": {
            "pixels": ["2", 0],
            "vae": ["5", 2],
            "mask": mask_terminal,
            "grow_mask_by": params.grow_mask_by,
        },
        "_meta": {"title": "Inpaint Latent"},
    }
    nodes["9"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["5", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["8", 0],
            "seed": params.seed,
            "steps": params.steps,
            "cfg": params.cfg,
            "sampler_name": params.sampler_name,
            "scheduler": params.scheduler,
            "denoise": params.denoise,
        },
        "_meta": {"title": "SEED"},
    }
    nodes["10"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["9", 0], "vae": ["5", 2]},
    }
    nodes["11"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["10", 0], "filename_prefix": params.output_prefix},
        "_meta": {"title": "Save"},
    }

    return nodes
