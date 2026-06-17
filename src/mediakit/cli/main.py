import typer

from mediakit.cli.commands import (
    bg_remove,
    compress,
    convert,
    img2video,
    img_edit,
    lqip,
    resize,
    txt2img,
    txt2video,
    upscale,
    variants,
)

app = typer.Typer(
    name="mediakit",
    help="Local image toolkit — compress, resize, convert, generate.",
    no_args_is_help=True,
)

app.add_typer(compress.app, name="compress")
app.add_typer(resize.app, name="resize")
app.add_typer(convert.app, name="convert")
app.add_typer(variants.app, name="variants")
app.add_typer(lqip.app, name="lqip")
app.add_typer(bg_remove.app, name="bg-remove")
app.add_typer(upscale.app, name="upscale")
app.add_typer(txt2img.app, name="txt2img")
app.add_typer(img_edit.app, name="img-edit")
app.add_typer(txt2video.app, name="txt2video")
app.add_typer(img2video.app, name="img2video")


@app.command("serve")
def serve_cmd() -> None:
    """Start the HTTP server (same as mediakit-server entrypoint)."""
    from mediakit.server.app import main as _main
    _main()


@app.command("worker")
def worker_cmd() -> None:
    """Start the arq worker (same as mediakit-worker entrypoint)."""
    from mediakit.jobs.worker import main as _main
    _main()


def main() -> None:
    app()
