import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.resize import resize as resize_op
from mediakit.schemas.ops import ImageFormat, ResizeMode, ResizeParams

app = typer.Typer(help="Resize an image (fit|fill|smart-crop|pad).")


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    width: Annotated[int, typer.Option("--width", "-w")],
    height: Annotated[int, typer.Option("--height", "-h")],
    mode: Annotated[ResizeMode, typer.Option("--mode", "-m")] = ResizeMode.fit,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    format: Annotated[ImageFormat | None, typer.Option("--format", "-f")] = None,
) -> None:
    params = ResizeParams(
        input=input, output=output, width=width, height=height, mode=mode, format=format
    )
    result = asyncio.run(resize_op(params))
    typer.echo(f"{result.width}×{result.height}  {result.output}")
