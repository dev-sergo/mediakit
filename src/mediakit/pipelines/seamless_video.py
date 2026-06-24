"""seamless_video pipeline.

Produces a clip of arbitrary length while hiding the seam that appears when a
video model is pushed past its native window.

Strategy (native window + continuation):
  1. If the requested length fits the model's native window, generate it in a
     single pass — there is no seam to hide. CogVideoX-5B gives ~6 s here
     (49 frames @ 8 fps), vs LTX-Video's ~2 s.
  2. Otherwise split into overlapping segments. Each next segment is generated
     with img2video seeded on the *last frame* of the previous one, so motion
     genuinely continues; a short crossfade over the overlap removes any residual
     micro-jump. Models without img2video (Wan) fall back to independent segments
     joined by the same crossfade — the cut is hidden but motion is not continued.

This is model-agnostic: it works with ltxv, cogvideox, and (crossfade-only) wan.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, cast

import structlog

from mediakit.backends.comfyui.video_models import PROFILES
from mediakit.backends.native.video import crossfade_concat, extract_last_frame
from mediakit.ops.img2video import img2video
from mediakit.ops.txt2video import txt2video
from mediakit.pipelines.base import BasePipeline, PipelineResult
from mediakit.schemas.video_ops import Img2VideoParams, Txt2VideoParams

log = structlog.get_logger(__name__)

I2vModel = Literal["ltxv", "cogvideox"]


class SeamlessVideoPipeline(BasePipeline):
    name = "seamless_video"

    async def run(  # type: ignore[override]
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        output_dir: Path,
        model: Literal["ltxv", "wan", "cogvideox"] = "cogvideox",
        input: Path | None = None,           # optional first-frame image (img2video start)
        total_frames: int = 97,
        fps: float | None = None,            # default: model-native fps
        width: int | None = None,
        height: int | None = None,
        overlap_frames: int = 8,
        steps: int = 50,
        cfg: float = 6.0,
        seed: int = -1,
        **_: Any,
    ) -> PipelineResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        profile = PROFILES[model]
        eff_fps = fps if fps is not None else profile.native_fps
        w = width if width is not None else profile.default_width
        h = height if height is not None else profile.default_height
        seg_len = profile.clamp_length(profile.native_max_frames)
        can_continue = profile.supports_img2video
        # An input frame can only be honored by models with an img2video path.
        start_with_image = input is not None and can_continue
        if input is not None and not can_continue:
            log.warning("seamless_video.img2video_unsupported", model=model)
        i2v_model = cast(I2vModel, model)

        def seg_seed(i: int) -> int:
            return seed if seed == -1 else seed + i

        # ── Single window: no stitching needed ───────────────────────────────
        if total_frames <= seg_len:
            length = profile.clamp_length(total_frames)
            final = output_dir / "video.mp4"
            if start_with_image:
                assert input is not None
                vid = await img2video(Img2VideoParams(
                    input=input, prompt=prompt, negative_prompt=negative_prompt,
                    model=i2v_model, width=w, height=h, length=length, fps=eff_fps,
                    steps=steps, cfg=cfg, seed=seed, output=final,
                ))
            else:
                vid = await txt2video(Txt2VideoParams(
                    prompt=prompt, negative_prompt=negative_prompt, model=model,
                    width=w, height=h, length=length, fps=eff_fps,
                    steps=steps, cfg=cfg, seed=seed, output=final,
                ))
            log.info("seamless_video.single_window", frames=length, seam_free=True)
            return PipelineResult(
                outputs=[vid.output],
                meta={
                    "segments": 1, "seam_free": True, "seed": vid.seed,
                    "duration_s": vid.duration_s, "model": model, "fps": eff_fps,
                },
            )

        # ── Multi-segment: continuation + crossfade ──────────────────────────
        overlap = max(1, min(overlap_frames, seg_len - 1))
        new_per_seg = seg_len - overlap
        n_segments = 1 + math.ceil((total_frames - seg_len) / new_per_seg)
        log.info(
            "seamless_video.plan",
            model=model, total_frames=total_frames, seg_len=seg_len,
            overlap=overlap, segments=n_segments, continuation=can_continue,
        )

        segments: list[Path] = []
        seeds: list[int] = []

        # Segment 0
        seg0 = output_dir / "seg_000.mp4"
        if start_with_image:
            assert input is not None
            v0 = await img2video(Img2VideoParams(
                input=input, prompt=prompt, negative_prompt=negative_prompt,
                model=i2v_model, width=w, height=h, length=seg_len, fps=eff_fps,
                steps=steps, cfg=cfg, seed=seg_seed(0), output=seg0,
            ))
        else:
            v0 = await txt2video(Txt2VideoParams(
                prompt=prompt, negative_prompt=negative_prompt, model=model,
                width=w, height=h, length=seg_len, fps=eff_fps,
                steps=steps, cfg=cfg, seed=seg_seed(0), output=seg0,
            ))
        segments.append(v0.output)
        seeds.append(v0.seed)

        # Continuation segments
        for i in range(1, n_segments):
            seg_path = output_dir / f"seg_{i:03d}.mp4"
            if can_continue:
                seed_frame = await extract_last_frame(
                    segments[-1], output_dir / f"seed_{i:03d}.png"
                )
                vi = await img2video(Img2VideoParams(
                    input=seed_frame, prompt=prompt, negative_prompt=negative_prompt,
                    model=i2v_model, width=w, height=h, length=seg_len, fps=eff_fps,
                    steps=steps, cfg=cfg, seed=seg_seed(i), output=seg_path,
                ))
            else:
                vi = await txt2video(Txt2VideoParams(
                    prompt=prompt, negative_prompt=negative_prompt, model=model,
                    width=w, height=h, length=seg_len, fps=eff_fps,
                    steps=steps, cfg=cfg, seed=seg_seed(i), output=seg_path,
                ))
            segments.append(vi.output)
            seeds.append(vi.seed)

        # Stitch with a crossfade over the overlap
        final = output_dir / "video.mp4"
        await crossfade_concat(
            segments, output=final, fps=eff_fps, overlap_s=overlap / eff_fps,
        )

        stitched_frames = seg_len * n_segments - overlap * (n_segments - 1)
        duration_s = round(stitched_frames / eff_fps, 2)
        log.info(
            "seamless_video.done",
            segments=n_segments, frames=stitched_frames,
            duration_s=duration_s, continuation=can_continue,
        )
        return PipelineResult(
            outputs=[final, *segments],
            meta={
                "segments": n_segments,
                "seam_free": False,
                "continuation": can_continue,
                "seeds": seeds,
                "duration_s": duration_s,
                "model": model,
                "fps": eff_fps,
            },
        )
