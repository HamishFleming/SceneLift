from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class ImageUpscaler(Protocol):
    name: str

    def upscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a BGR frame at a larger resolution."""


@dataclass(frozen=True)
class UpscalerSpec:
    key: str
    display_name: str
    kind: str
    scale: int
    local_weights: Path | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
