"""product_shot pipeline.

bg_remove → contact_shadow (native Pillow) → composite → [upscale →] variants

E-commerce product photo: cut out subject, add a contact shadow (blurred oval
at the base of the object), composite onto a clean gradient background.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from PIL import Image, ImageDraw, ImageFilter

from mediakit.ops.bg_remove import bg_remove
from mediakit.ops.upscale import upscale
from mediakit.ops.variants import variants as make_variants
from mediakit.pipelines.base import BasePipeline, PipelineResult
from mediakit.schemas.ai_ops import BgRemoveParams, BiRefNetModel, UpscaleModel, UpscaleParams
from mediakit.schemas.ops import ImageFormat, Quality, VariantsParams

log = structlog.get_logger(__name__)

_DEFAULT_WIDTHS = [640, 1024, 1280]
_DEFAULT_FORMATS = [ImageFormat.webp]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _make_gradient_bg(
    size: tuple[int, int],
    color: tuple[int, int, int],
    strength: float,
) -> Image.Image:
    """Radial vignette: lighter center, slightly darker edges."""
    small = 64
    sm = Image.new("L", (small, small), 0)
    pix = sm.load()
    cx, cy = small // 2, small // 2
    for y in range(small):
        for x in range(small):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            d_max = (cx**2 + cy**2) ** 0.5
            if pix is not None:
                pix[x, y] = int(min(d / d_max * 255 * strength * 2, 255))
    vignette = sm.resize(size, Image.Resampling.BILINEAR)
    bg = Image.new("RGB", size, color)
    edge = (max(0, color[0] - 50), max(0, color[1] - 50), max(0, color[2] - 50))
    dark = Image.new("RGB", size, edge)
    bg.paste(dark, mask=vignette)
    return bg


def _make_contact_shadow(
    nobg: Image.Image,
    shadow_opacity: int = 90,
    blur_radius: int = 18,
) -> Image.Image:
    """Blurred oval contact shadow fitted to the base of the subject.

    Draws an ellipse whose width matches the object's bounding box and whose
    height is ~12% of the object height, positioned just below the bottom edge.
    """
    alpha = nobg.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return Image.new("RGBA", nobg.size, (0, 0, 0, 0))

    left, _top, right, bottom = bbox
    obj_w = right - left
    obj_h = bottom - _top

    # Oval dimensions
    ow = int(obj_w * 0.75)
    oh = int(obj_h * 0.12)
    oh = max(oh, 8)

    cx = (left + right) // 2
    cy = bottom

    shadow_layer = Image.new("L", nobg.size, 0)
    draw = ImageDraw.Draw(shadow_layer)
    draw.ellipse(
        [cx - ow // 2, cy - oh // 2, cx + ow // 2, cy + oh // 2],
        fill=shadow_opacity,
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur_radius))

    result = Image.new("RGBA", nobg.size, (0, 0, 0, 0))
    dark = Image.new("RGB", nobg.size, (30, 30, 30))
    result.paste(dark, mask=shadow_layer)
    return result


def _composite(
    nobg: Path,
    output: Path,
    bg_color: tuple[int, int, int],
    padding_pct: float,
    gradient_strength: float,
    shadow_opacity: int,
    shadow_blur: int,
) -> None:
    """Composite subject + contact shadow onto a gradient background."""
    img = Image.open(nobg).convert("RGBA")
    w, h = img.size
    pad = int(max(w, h) * padding_pct)
    cw, ch = w + pad * 2, h + pad * 2

    canvas = _make_gradient_bg((cw, ch), bg_color, gradient_strength).convert("RGBA")

    # Contact shadow sits inside the padded area, aligned with the subject
    shadow = _make_contact_shadow(img, shadow_opacity=shadow_opacity, blur_radius=shadow_blur)
    shadow_canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    shadow_canvas.paste(shadow, (pad, pad))
    canvas = Image.alpha_composite(canvas, shadow_canvas)

    # Place subject on top
    subj_canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    subj_canvas.paste(img, (pad, pad), mask=img)
    Image.alpha_composite(canvas, subj_canvas).convert("RGB").save(output)


class ProductShotPipeline(BasePipeline):
    name = "product_shot"

    async def run(  # type: ignore[override]
        self,
        *,
        input: Path,
        output_dir: Path | None = None,
        birefnet_model: BiRefNetModel = BiRefNetModel.hr,
        bg_color: str = "#FFFFFF",
        padding_pct: float = 0.1,
        gradient_strength: float = 0.12,
        shadow_opacity: int = 90,
        shadow_blur: int = 18,
        do_upscale: bool = True,
        upscale_model: UpscaleModel = UpscaleModel.nmkd,
        upscale_scale: float = 2.0,
        formats: list[ImageFormat] | None = None,
        widths: list[int] | None = None,
        quality: Quality = Quality.high,
        **_: Any,
    ) -> PipelineResult:
        out_dir = output_dir or input.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input.stem
        outputs: list[Path] = []

        # 1. Remove background → transparent PNG
        log.info("product_shot.bg_remove", input=str(input))
        bg = await bg_remove(
            BgRemoveParams(
                input=input,
                output=out_dir / f"{stem}_nobg.png",
                model=birefnet_model,
                background_mode="transparent",
            )
        )
        outputs.append(bg.output)

        # 2. Composite: gradient bg + contact shadow + subject
        bg_png = out_dir / f"{stem}_bg.png"
        log.info("product_shot.composite", color=bg_color, shadow_opacity=shadow_opacity)
        _composite(
            bg.output,
            bg_png,
            _hex_to_rgb(bg_color),
            padding_pct,
            gradient_strength,
            shadow_opacity,
            shadow_blur,
        )
        outputs.append(bg_png)
        current = bg_png

        # 3. Optional upscale
        if do_upscale:
            log.info("product_shot.upscale", scale=upscale_scale)
            up = await upscale(
                UpscaleParams(
                    input=current,
                    output=out_dir / f"{stem}_upscaled.png",
                    model=upscale_model,
                    scale=upscale_scale,
                )
            )
            current = up.output
            outputs.append(current)

        # 4. Responsive variant set
        log.info("product_shot.variants")
        vresult = await make_variants(
            VariantsParams(
                input=current,
                output_dir=out_dir,
                widths=widths or _DEFAULT_WIDTHS,
                formats=formats or _DEFAULT_FORMATS,
                quality=quality,
                stem=stem,
            )
        )
        outputs.extend(v.path for v in vresult.variants)

        log.info("product_shot.done", files=len(outputs), variants=len(vresult.variants))
        return PipelineResult(outputs=outputs, meta={"variants": len(vresult.variants)})
