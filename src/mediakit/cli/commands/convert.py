import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.convert import convert as convert_op
from mediakit.schemas.ops import ConvertParams, ImageFormat, Quality, parse_quality

app = typer.Typer(help="Convert image to a different format.")


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    format: Annotated[ImageFormat, typer.Option("--format", "-f", help="Output format")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    quality: Annotated[str, typer.Option("--quality", "-q")] = "high",
) -> None:
    q: Quality | int = parse_quality(quality)
    params = ConvertParams(input=input, output=output, format=format, quality=q)
    result = asyncio.run(convert_op(params))
    typer.echo(f"{result.input_bytes // 1024}KB → {result.output_bytes // 1024}KB  {result.output}")
