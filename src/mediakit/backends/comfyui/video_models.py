"""Static capability profiles for the video generation models.

Single source of truth for "how long a seamless clip can each model produce in a
single pass". Used by the seamless_video pipeline to decide whether a request
fits in one window (no stitching) or must be split into continued segments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoModelProfile:
    native_fps: float  # fps the model was trained at — correct playback rate
    native_max_frames: int  # longest seam-free window in a single pass
    frame_step: int  # valid frame counts are frame_base + frame_step * n
    frame_base: int
    supports_img2video: bool  # required for last-frame continuation stitching
    default_width: int
    default_height: int

    def clamp_length(self, frames: int) -> int:
        """Round a frame count down to the nearest model-valid value within the window."""
        capped = min(frames, self.native_max_frames)
        n = max(0, (capped - self.frame_base) // self.frame_step)
        return self.frame_base + self.frame_step * n


# LTX-Video and Wan windows mirror the limits documented in docs/DOCS.md.
PROFILES: dict[str, VideoModelProfile] = {
    "ltxv": VideoModelProfile(
        native_fps=24.0,
        native_max_frames=49,
        frame_step=8,
        frame_base=9,
        supports_img2video=True,
        default_width=768,
        default_height=512,
    ),
    "wan": VideoModelProfile(
        native_fps=16.0,
        native_max_frames=81,
        frame_step=4,
        frame_base=1,
        supports_img2video=False,
        default_width=832,
        default_height=480,
    ),
    "cogvideox": VideoModelProfile(
        native_fps=8.0,
        native_max_frames=49,
        frame_step=8,
        frame_base=1,
        supports_img2video=True,
        default_width=720,
        default_height=480,
    ),
}
