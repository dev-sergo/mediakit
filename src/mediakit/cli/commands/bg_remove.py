import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.bg_remove import bg_remove as bg_remove_op
from mediakit.schemas.ai_ops import BgRemoveParams, BiRefNetModel

app = typer.Typer(help="Remove background via BiRefNet (requires ComfyUI running).")


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    model: Annotated[BiRefNetModel, typer.Option("--model", "-m")] = BiRefNetModel.hr,
    background: Annotated[str, typer.Option("--bg", help="transparent|color")] = "transparent",
    color: Annotated[str, typer.Option("--color", help="Background HEX color")] = "#FFFFFF",
    blur: Annotated[int, typer.Option("--blur")] = 0,
) -> None:
    params = BgRemoveParams(
        input=input,
        output=output,
        model=model,
        background_mode=background,  # type: ignore[arg-type]
        background_color=color,
        mask_blur=blur,
    )
    result = asyncio.run(bg_remove_op(params))
    typer.echo(f"Output: {result.output}")
