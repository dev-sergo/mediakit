"""Qwen Image Edit 2511 workflow builder (instruction-based image editing).

Architecture (ComfyUI-RMBG v3.0.0 nodes):
  UNETLoader (qwen_image_edit UNet) → model
  CLIPLoader (qwen_2.5_vl, type=qwen_vl) → clip
  VAELoader (qwen_image_vae) → vae
  LoadImage → input image
  TextEncodeQwenImageEdit (prompt + image) → conditioning
  BiRefNetRMBG (background mask) → mask
  EmptyQwenImageLayeredLatentImage (image + mask) → latent
  LoraLoader (Lightning 4-step LoRA, optional) → patched model
  KSampler (4 steps, euler, simple) → latent
  VAEDecode → image
  ImageCompositeMasked (blend edit with original) → composite
  SaveImage → output

Qwen vs SDXL inpainting:
  Qwen understands instruction-based prompts ("replace the background with...")
  SDXL uses diffusion inpainting (less semantic understanding)
  Qwen Lightning: 4 steps only, much faster, slight quality tradeoff

VERIFY on GPU box (ComfyUI-RMBG v3.0.0 node names may differ):
  - TextEncodeQwenImageEdit: check exact class name in ComfyUI
  - EmptyQwenImageLayeredLatentImage: check class name
  - ImageCompositeMasked: standard ComfyUI node
  - LoraLoader: standard ComfyUI node
  - qwen_image_vae.safetensors: correct VAE path
  - Lightning LoRA path: loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

WorkflowDict = dict[str, Any]

_QWEN_UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
_QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
_QWEN_VAE = "qwen_image_vae.safetensors"
_QWEN_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


@dataclass(frozen=True)
class ImgEditQwenWorkflowParams:
    positive_prompt: str
    image_filename: str
    negative_prompt: str = ""
    unet_name: str = _QWEN_UNET
    clip_name: str = _QWEN_CLIP
    vae_name: str = _QWEN_VAE
    lora_name: str = _QWEN_LORA
    lora_strength: float = 1.0      # 0.0 = no LoRA (base Qwen), 1.0 = full Lightning
    use_lora: bool = True           # Lightning LoRA: 4 steps; base Qwen: 20+ steps
    steps: int = 4                  # 4 for Lightning, 20+ for base
    cfg: float = 1.0                # Qwen uses very low CFG
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    width: int = 1024
    height: int = 1024
    output_prefix: str = "qwen_edit"


def build_img_edit_qwen_workflow(params: ImgEditQwenWorkflowParams) -> WorkflowDict:
    nodes: dict[str, dict[str, Any]] = {}

    # 1. Load Qwen UNet (diffusion_models/ format)
    nodes["1"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": params.unet_name, "weight_dtype": "fp8_e4m3fn"},
        "_meta": {"title": "Qwen UNet Loader"},
    }

    # 2. Load Qwen text encoder (type=qwen_image confirmed valid in ComfyUI 0.18)
    nodes["2"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": params.clip_name, "type": "qwen_image"},
        "_meta": {"title": "Qwen CLIP Loader"},
    }

    # 3. Load Qwen VAE
    nodes["3"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": params.vae_name},
        "_meta": {"title": "Qwen VAE Loader"},
    }

    # 4. Load input image (uploaded to ComfyUI server)
    nodes["4"] = {
        "class_type": "LoadImage",
        "inputs": {"image": params.image_filename},
        "_meta": {"title": "Input Image"},
    }

    # 5. Apply Lightning LoRA (optional — skip if use_lora=False)
    if params.use_lora and params.lora_strength > 0:
        nodes["5"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["2", 0],
                "lora_name": params.lora_name,
                "strength_model": params.lora_strength,
                "strength_clip": params.lora_strength,
            },
            "_meta": {"title": "Lightning LoRA"},
        }
        model_ref = ["5", 0]
        clip_ref = ["5", 1]
    else:
        model_ref = ["1", 0]
        clip_ref = ["2", 0]

    # 6. Encode prompt + image into Qwen conditioning (input key is "prompt" not "text")
    nodes["6"] = {
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {
            "clip": clip_ref,
            "image": ["4", 0],
            "prompt": params.positive_prompt,
        },
        "_meta": {"title": "PROMPT_POSITIVE"},
    }

    # 7. Extract foreground via BiRefNet (background="Alpha" → RGBA, alpha=foreground mask)
    nodes["7"] = {
        "class_type": "BiRefNetRMBG",
        "inputs": {
            "image": ["4", 0],
            "model": "BiRefNet-HR",
            "mask_blur": 0,
            "mask_offset": 0,
            "invert_output": False,
            "refine_foreground": True,
            "background": "Alpha",         # transparent bg → alpha channel = fg mask
            "background_color": "#FFFFFF",
        },
        "_meta": {"title": "BiRefNet (alpha mask)"},
    }

    # 7b. Extract alpha channel → MASK (1=foreground, 0=background)
    nodes["7b"] = {
        "class_type": "ImageToMask",
        "inputs": {"image": ["7", 0], "channel": "alpha"},
    }

    # 7c. Invert: 0=foreground(keep), 1=background(replace with generated)
    nodes["7c"] = {
        "class_type": "InvertMask",
        "inputs": {"mask": ["7b", 0]},
        "_meta": {"title": "Background Mask"},
    }

    # 8. Create empty layered latent (width/height/batch_size/layers — NOT image/mask/vae)
    nodes["8"] = {
        "class_type": "EmptyQwenImageLayeredLatentImage",
        "inputs": {
            "width": params.width,
            "height": params.height,
            "batch_size": 1,
            "layers": 1,
        },
        "_meta": {"title": "Layered Latent"},
    }

    # 9. KSampler — 4 steps for Lightning, cfg=1.0
    nodes["9"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["6", 0],
            "negative": ["6", 0],   # Qwen ignores negative — use same as positive
            "latent_image": ["8", 0],
            "seed": params.seed,
            "steps": params.steps,
            "cfg": params.cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
        "_meta": {"title": "SEED"},
    }

    # 10. Decode edited region
    nodes["10"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["9", 0], "vae": ["3", 0]},
    }

    # 11. Composite: blend edited region with original using MASK type
    nodes["11"] = {
        "class_type": "ImageCompositeMasked",
        "inputs": {
            "destination": ["4", 0],   # original image
            "source": ["10", 0],       # edited region
            "mask": ["7c", 0],         # inverted background mask (MASK type)
            "x": 0,
            "y": 0,
            "resize_source": False,
        },
        "_meta": {"title": "Composite"},
    }

    # 12. Save output
    nodes["12"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["11", 0],
            "filename_prefix": params.output_prefix,
        },
        "_meta": {"title": "Save"},
    }

    return nodes
