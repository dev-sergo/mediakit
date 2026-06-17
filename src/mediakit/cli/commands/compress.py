import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.compress import compress as compress_op
from mediakit.schemas.ops import CompressParams, ImageFormat, Quality, parse_quality

app = typer.Typer(help="Compress an image (JPEG/WebP/AVIF/PNG).")


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    format: Annotated[ImageFormat, typer.Option("--format", "-f")] = ImageFormat.webp,
    quality: Annotated[
        str, typer.Option("--quality", "-q", help="low|medium|high|max or 0-100")
    ] = "high",
    max_width: Annotated[int | None, typer.Option("--max-width")] = None,
    no_strip: Annotated[bool, typer.Option("--no-strip", help="Keep EXIF metadata")] = False,
) -> None:
    q: Quality | int = parse_quality(quality)
    params = CompressParams(
        input=input,
        output=output,
        format=format,
        quality=q,
        max_width=max_width,
        strip_metadata=not no_strip,
    )
    result = asyncio.run(compress_op(params))
    typer.echo(
        f"{result.input_bytes // 1024}KB → {result.output_bytes // 1024}KB "
        f"({result.savings_pct:+.1f}%)  {result.output}"
    )
