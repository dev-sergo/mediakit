import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.variants import variants as variants_op
from mediakit.schemas.ops import ImageFormat, Quality, VariantsParams, parse_quality

app = typer.Typer(help="Generate responsive image variants at multiple widths.")


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    sizes: Annotated[
        str, typer.Option("--sizes", "-s", help="Comma-separated widths")
    ] = "640,768,1024,1280,1536",
    formats: Annotated[
        str, typer.Option("--formats", "-f", help="Comma-separated formats")
    ] = "webp",
    quality: Annotated[str, typer.Option("--quality", "-q")] = "high",
    output_dir: Annotated[Path | None, typer.Option("--out-dir")] = None,
    stem: Annotated[str | None, typer.Option("--stem")] = None,
) -> None:
    widths = [int(x.strip()) for x in sizes.split(",")]
    fmts = [ImageFormat(x.strip()) for x in formats.split(",")]
    q: Quality | int = parse_quality(quality)
    params = VariantsParams(
        input=input, output_dir=output_dir, widths=widths, formats=fmts, quality=q, stem=stem
    )
    result = asyncio.run(variants_op(params))
    for v in result.variants:
        typer.echo(f"  {v.width}×{v.height} {v.format.value}  {v.path}")
    typer.echo(f"{len(result.variants)} variants generated")
