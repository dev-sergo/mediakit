"""Programmatic builder for the upscale ComfyUI workflow.

LoadImage → UpscaleModelLoader → ImageUpscaleWithModel (4×) → [ImageScaleBy?] → SaveImage

ESRGAN models always produce 4×. Lower target scales are achieved by
lanczos-downscaling after — preserves the detail-recovery benefit of the
upscaler while landing on a sensible final size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class UpscaleWorkflowParams:
    image_filename: str
    upscale_model: str = "4x_NMKD-Siax_200k.pth"
    target_scale: float = 2.0  # 1.5..4.0; model always outputs 4×, then scaled down
    output_prefix: str = "upscale"


def build_upscale_workflow(params: UpscaleWorkflowParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "LoadImage",
        "inputs": {"image": params.image_filename},
        "_meta": {"title": "INPUT_IMAGE"},
    }
    nodes["2"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": params.upscale_model},
    }
    nodes["3"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
        "_meta": {"title": "4× Upscale"},
    }

    final: list[Any] = ["3", 0]

    if params.target_scale < 4.0:
        scale_factor = max(params.target_scale / 4.0, 0.25)
        nodes["4"] = {
            "class_type": "ImageScaleBy",
            "inputs": {"image": ["3", 0], "upscale_method": "lanczos", "scale_by": scale_factor},
            "_meta": {"title": f"Down to {params.target_scale}×"},
        }
        final = ["4", 0]

    nodes["5"] = {
        "class_type": "SaveImage",
        "inputs": {"images": final, "filename_prefix": params.output_prefix},
        "_meta": {"title": "Save"},
    }

    return nodes
