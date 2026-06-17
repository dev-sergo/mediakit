import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.lqip import lqip as lqip_op
from mediakit.schemas.ops import LqipParams

app = typer.Typer(help="Generate blur placeholder (base64 WebP data URL).")


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    size: Annotated[int, typer.Option("--size", help="Longest edge in pixels")] = 16,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write to file instead of stdout")
    ] = None,
) -> None:
    params = LqipParams(input=input, size=size)
    result = asyncio.run(lqip_op(params))
    if output:
        output.write_text(result.data_url)
        typer.echo(f"Written to {output}")
    else:
        typer.echo(result.data_url)
