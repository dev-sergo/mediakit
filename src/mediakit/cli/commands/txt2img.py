import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.txt2img import txt2img as txt2img_op
from mediakit.schemas.ai_ops import Txt2ImgParams

app = typer.Typer(help="Generate image from text prompt (SDXL or Flux 2, requires ComfyUI).")


@app.callback(invoke_without_command=True)
def cmd(
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Positive prompt")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    backend: Annotated[str, typer.Option("--backend", "-b", help="sdxl | flux")] = "sdxl",
    negative: Annotated[str, typer.Option("--negative", "-n")] = "",
    width: Annotated[int, typer.Option("--width", "-w")] = 1024,
    height: Annotated[int, typer.Option("--height", "-h")] = 1024,
    steps: Annotated[int, typer.Option("--steps")] = 25,
    cfg: Annotated[
        float, typer.Option("--cfg", help="CFG (SDXL) or guidance (Flux, use 2-5)")
    ] = 7.5,
    seed: Annotated[int, typer.Option("--seed", help="-1 for random")] = -1,
    checkpoint: Annotated[str, typer.Option("--ckpt")] = "RealVisXL_V5.0_inpainting.safetensors",
) -> None:
    import secrets

    actual_seed = secrets.randbits(32) if seed == -1 else seed
    params = Txt2ImgParams(
        prompt=prompt,
        negative_prompt=negative,
        backend=backend,  # type: ignore[arg-type]
        checkpoint=checkpoint,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=actual_seed,
        output=output,
    )
    result = asyncio.run(txt2img_op(params))
    typer.echo(f"seed={result.seed}  {result.output}")
