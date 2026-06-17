"""Base class for named pipelines — compositions of ops."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class PipelineResult(BaseModel):
    outputs: list[Path]
    meta: dict[str, Any] = {}


class BasePipeline(ABC):
    name: str  # registry key, e.g. "article_cover"

    @abstractmethod
    async def run(self, **kwargs: Any) -> PipelineResult:
        ...
