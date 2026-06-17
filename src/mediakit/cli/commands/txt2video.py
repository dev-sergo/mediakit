import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mediakit.ops.txt2video import txt2video as txt2video_op
from mediakit.schemas.video_ops import Txt2VideoParams

_HELP = "Generate video from text prompt via LTX-Video or Wan 2.1 (requires ComfyUI)."
app = typer.Typer(help=_HELP)


@app.callback(invoke_without_command=True)
def cmd(
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Positive prompt")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="ltxv | wan")] = "ltxv",
    negative: Annotated[str, typer.Option("--negative", "-n")] = "",
    width: Annotated[int, typer.Option("--width", "-w")] = 768,
    height: Annotated[int, typer.Option("--height", "-h")] = 512,
    length: Annotated[int, typer.Option("--length", help="Frames (LTX: 9+8n, Wan: 4n+1)")] = 49,
    fps: Annotated[float, typer.Option("--fps")] = 24.0,
    steps: Annotated[int, typer.Option("--steps")] = 15,
    cfg: Annotated[float, typer.Option("--cfg", help="Guidance (2–5 for video models)")] = 2.5,
    seed: Annotated[int, typer.Option("--seed", help="-1 for random")] = -1,
) -> None:
    params = Txt2VideoParams(
        prompt=prompt,
        negative_prompt=negative,
        model=model,  # type: ignore[arg-type]
        width=width,
        height=height,
        length=length,
        fps=fps,
        steps=steps,
        cfg=cfg,
        seed=seed,
        output=output,
    )
    result = asyncio.run(txt2video_op(params))
    typer.echo(f"seed={result.seed}  duration={result.duration_s}s  {result.output}")
