import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.upscale import upscale as upscale_op
from mediakit.schemas.ai_ops import UpscaleModel, UpscaleParams

app = typer.Typer(help="Upscale image via ESRGAN (requires ComfyUI running).")


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    model: Annotated[UpscaleModel, typer.Option("--model", "-m")] = UpscaleModel.nmkd,
    scale: Annotated[float, typer.Option("--scale", "-s", help="Target scale (1.5–4.0)")] = 2.0,
) -> None:
    params = UpscaleParams(input=input, output=output, model=model, scale=scale)
    result = asyncio.run(upscale_op(params))
    typer.echo(f"Output: {result.output}")
