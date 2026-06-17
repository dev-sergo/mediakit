import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.img_edit import img_edit as img_edit_op
from mediakit.schemas.ai_ops import ImgEditParams

_HELP = "Edit image background via SDXL inpainting or Qwen Image Edit (requires ComfyUI)."
app = typer.Typer(help=_HELP)


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Edit instruction or description")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    backend: Annotated[str, typer.Option("--backend", "-b", help="sdxl | qwen")] = "sdxl",
    negative: Annotated[str, typer.Option("--negative", "-n")] = "",
    width: Annotated[int, typer.Option("--width", "-w")] = 1024,
    height: Annotated[int, typer.Option("--height", "-h")] = 1024,
    steps: Annotated[int, typer.Option("--steps")] = 25,
    cfg: Annotated[float, typer.Option("--cfg")] = 7.5,
    seed: Annotated[int, typer.Option("--seed")] = -1,
    mask: Annotated[str, typer.Option("--mask", help="background|full")] = "background",
    lora_strength: Annotated[
        float, typer.Option("--lora-strength", help="Qwen Lightning LoRA weight (0-2)")
    ] = 1.0,
) -> None:
    import secrets
    actual_seed = secrets.randbits(32) if seed == -1 else seed
    params = ImgEditParams(
        input=input,
        prompt=prompt,
        negative_prompt=negative,
        backend=backend,  # type: ignore[arg-type]
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=actual_seed,
        mask_target=mask,  # type: ignore[arg-type]
        lora_strength=lora_strength,
        output=output,
    )
    result = asyncio.run(img_edit_op(params))
    typer.echo(f"seed={result.seed}  {result.output}")
