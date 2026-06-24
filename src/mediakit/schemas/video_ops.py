from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Txt2VideoParams(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model: Literal["ltxv", "wan", "cogvideox"] = "ltxv"
    width: int = Field(default=768, ge=256, le=1920, multiple_of=8)
    height: int = Field(default=512, ge=256, le=1920, multiple_of=8)
    # Number of frames. LTX: 9+8n (e.g. 25, 33, 49, 65, 97, 161).
    # Wan: multiples of 4+1 (e.g. 17, 33, 49, 65, 81).
    # CogVideoX-5B: 8n+1 (49 = native 6s window at 8 fps).
    length: int = Field(default=49, ge=9, le=257)
    fps: float = Field(default=24.0, ge=8.0, le=60.0)
    steps: int = Field(default=15, ge=1, le=50)
    cfg: float = Field(default=2.5, ge=1.0, le=15.0)
    seed: int = -1
    output: Path | None = None


class Img2VideoParams(BaseModel):
    input: Path
    prompt: str
    negative_prompt: str = ""
    # LTX-Video and CogVideoX-5B-I2V support img2vid; Wan phantom requires a subject image.
    model: Literal["ltxv", "cogvideox"] = "ltxv"
    width: int = Field(default=768, ge=256, le=1920, multiple_of=8)
    height: int = Field(default=512, ge=256, le=1920, multiple_of=8)
    length: int = Field(default=49, ge=9, le=257)
    fps: float = Field(default=24.0, ge=8.0, le=60.0)
    steps: int = Field(default=15, ge=1, le=50)
    cfg: float = Field(default=2.5, ge=1.0, le=15.0)
    seed: int = -1
    output: Path | None = None


class VideoResult(BaseModel):
    output: Path  # .mp4 file
    seed: int
    duration_s: float
