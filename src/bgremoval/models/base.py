from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class BackgroundRemover(Protocol):
    name: str

    def remove(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return an RGBA frame."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    kind: str
    huggingface_id: str | None = None
    local_weights: Path | None = None
    supports_video: bool = True
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

