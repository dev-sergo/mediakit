import secrets
import shutil
import uuid

import structlog

from mediakit.backends.comfyui.client import ComfyUIClient
from mediakit.backends.comfyui.workflows.txt2img import (
    Txt2ImgWorkflowParams,
    build_txt2img_workflow,
)
from mediakit.backends.comfyui.workflows.txt2img_flux import (
    Txt2ImgFluxWorkflowParams,
    build_txt2img_flux_workflow,
)
from mediakit.backends.native.metadata import write_metadata
from mediakit.config import settings
from mediakit.schemas.ai_ops import Txt2ImgParams, Txt2ImgResult

log = structlog.get_logger(__name__)


async def txt2img(params: Txt2ImgParams) -> Txt2ImgResult:
    seed = secrets.randbits(32) if params.seed == -1 else params.seed
    settings.ensure_storage_dirs()
    tmp_dir = settings.storage_outputs / "tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True)

    async with ComfyUIClient(
        settings.comfyui_url,
        output_dir=tmp_dir,
        timeout_seconds=float(settings.comfyui_timeout_s),
    ) as comfy:
        if params.backend == "flux":
            workflow = build_txt2img_flux_workflow(
                Txt2ImgFluxWorkflowParams(
                    positive_prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    steps=params.steps,
                    guidance=params.cfg,  # cfg field reused as Flux guidance
                    seed=seed,
                )
            )
        else:
            workflow = build_txt2img_workflow(
                Txt2ImgWorkflowParams(
                    positive_prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    checkpoint=params.checkpoint,
                    width=params.width,
                    height=params.height,
                    steps=params.steps,
                    cfg=params.cfg,
                    seed=seed,
                    sampler_name=params.sampler,
                    scheduler=params.scheduler,
                )
            )

        prompt_id = await comfy.submit_workflow(workflow)
        raw_outputs = await comfy.wait_for_result(prompt_id)

    final = params.output or (settings.storage_outputs / f"txt2img_{uuid.uuid4().hex}.png")
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_outputs[0], final)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    write_metadata(final, seed=seed, steps=params.steps, cfg=params.cfg,
                   checkpoint=params.checkpoint, backend=params.backend,
                   prompt=params.prompt, width=params.width, height=params.height)
    log.info("txt2img.done", output=str(final), seed=seed, backend=params.backend)
    return Txt2ImgResult(output=final, seed=seed)
