"""Programmatic builder for the background_removal ComfyUI workflow.

LoadImage → [ImageScale?] → BiRefNetRMBG → SaveImage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WorkflowDict = dict[str, Any]


@dataclass(frozen=True)
class BgRemoveWorkflowParams:
    image_filename: str
    birefnet_model: str = "BiRefNet-HR"
    background_mode: str = "transparent"  # "transparent" | "color"
    background_color: str = "#FFFFFF"
    mask_blur: int = 0
    mask_offset: int = 0
    refine_foreground: bool = True
    width: int = 0
    height: int = 0
    output_prefix: str = "bg_remove"


def build_bg_remove_workflow(params: BgRemoveWorkflowParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    nodes["1"] = {
        "class_type": "LoadImage",
        "inputs": {"image": params.image_filename},
        "_meta": {"title": "INPUT_IMAGE"},
    }

    source: list[Any] = ["1", 0]

    if params.width > 0 and params.height > 0:
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
        source = ["2", 0]

    bg_mode = "Alpha" if params.background_mode == "transparent" else "Color"

    nodes["3"] = {
        "class_type": "BiRefNetRMBG",
        "inputs": {
            "image": source,
            "model": params.birefnet_model,
            "mask_blur": params.mask_blur,
            "mask_offset": params.mask_offset,
            "invert_output": False,
            "refine_foreground": params.refine_foreground,
            "background": bg_mode,
            "background_color": params.background_color,
        },
        "_meta": {"title": "BiRefNet"},
    }

    nodes["4"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["3", 0], "filename_prefix": params.output_prefix},
        "_meta": {"title": "Save"},
    }

    return nodes
