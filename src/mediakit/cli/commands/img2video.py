import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.img2video import img2video as img2video_op
from mediakit.schemas.video_ops import Img2VideoParams

app = typer.Typer(
    help="Animate an image into video via LTX-Video or CogVideoX img2vid (requires ComfyUI)."
)


@app.callback(invoke_without_command=True)
def cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input image path")],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Motion description prompt")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="ltxv | cogvideox")] = "ltxv",
    negative: Annotated[str, typer.Option("--negative", "-n")] = "",
    width: Annotated[int, typer.Option("--width", "-w")] = 768,
    height: Annotated[int, typer.Option("--height", "-h")] = 512,
    length: Annotated[int, typer.Option("--length")] = 49,
    fps: Annotated[float, typer.Option("--fps")] = 24.0,
    steps: Annotated[int, typer.Option("--steps")] = 15,
    cfg: Annotated[float, typer.Option("--cfg")] = 2.0,
    seed: Annotated[int, typer.Option("--seed", help="-1 for random")] = -1,
) -> None:
    params = Img2VideoParams(
        input=input,
        prompt=prompt,
        model=model,  # type: ignore[arg-type]
        negative_prompt=negative,
        width=width,
        height=height,
        length=length,
        fps=fps,
        steps=steps,
        cfg=cfg,
        seed=seed,
        output=output,
    )
    result = asyncio.run(img2video_op(params))
    typer.echo(f"seed={result.seed}  duration={result.duration_s}s  {result.output}")
