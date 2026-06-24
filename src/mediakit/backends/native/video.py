"""Native video stitching helpers — frame extraction (OpenCV) and crossfade
concatenation (ffmpeg). Used by the seamless_video pipeline to join segments.

ffmpeg is assumed present (ComfyUI's VideoHelperSuite already depends on it).
OpenCV (opencv-python-headless) is a hard dependency of mediakit.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import structlog

log = structlog.get_logger(__name__)


class FfmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe subprocess exits non-zero."""


def _read_last_frame_sync(video: Path, out: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise FfmpegError(f"OpenCV could not open {video}")
        last = None
        # CAP_PROP_FRAME_COUNT is unreliable across codecs; iterate to be safe.
        # Segments are short (≤ ~100 frames) so this is cheap.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
        if last is None:
            raise FfmpegError(f"No decodable frames in {video}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), last):
            raise FfmpegError(f"OpenCV could not write {out}")
        return out
    finally:
        cap.release()


async def extract_last_frame(video: Path, out: Path) -> Path:
    """Save the final frame of ``video`` to ``out`` (PNG). Continuation seed for
    the next segment."""
    return await asyncio.to_thread(_read_last_frame_sync, video, out)


async def _run(cmd: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise FfmpegError(
            f"{cmd[0]} exited {proc.returncode}: {stderr.decode(errors='replace')[-2000:]}"
        )


async def probe_duration(video: Path) -> float:
    """Return container duration in seconds via ffprobe."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise FfmpegError(f"ffprobe failed: {stderr.decode(errors='replace')}")
    return float(stdout.decode().strip())


async def crossfade_concat(
    segments: list[Path],
    *,
    output: Path,
    fps: float,
    overlap_s: float,
) -> Path:
    """Concatenate ``segments`` into ``output`` with an ``overlap_s`` crossfade at
    each join, using ffmpeg's xfade filter.

    With continuation segments (each starting on the previous last frame), the
    overlap blends near-identical frames, so the join is invisible; the same call
    also hides a hard cut between independently generated segments, just less
    cleanly. Re-encodes to H.264 / yuv420p for broad playback compatibility.
    """
    if not segments:
        raise ValueError("crossfade_concat requires at least one segment")
    if len(segments) == 1:
        # Nothing to blend — re-encode through to normalize the container.
        await _run([
            "ffmpeg", "-y", "-i", str(segments[0]),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(output),
        ])
        return output

    durations = [await probe_duration(s) for s in segments]

    cmd: list[str] = ["ffmpeg", "-y"]
    for seg in segments:
        cmd += ["-i", str(seg)]

    # Normalize timebase/fps/pixel format so xfade gets consistent streams.
    filters: list[str] = []
    for i in range(len(segments)):
        filters.append(
            f"[{i}:v]fps={fps},format=yuv420p,setpts=PTS-STARTPTS[s{i}]"
        )

    prev = "s0"
    elapsed = durations[0]
    for i in range(1, len(segments)):
        offset = elapsed - overlap_s * i
        label = f"v{i}" if i < len(segments) - 1 else "out"
        filters.append(
            f"[{prev}][s{i}]xfade=transition=fade:"
            f"duration={overlap_s}:offset={max(offset, 0):.3f}[{label}]"
        )
        elapsed += durations[i]
        prev = label

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    log.info("video.crossfade_concat", segments=len(segments), overlap_s=overlap_s)
    await _run(cmd)
    return output
