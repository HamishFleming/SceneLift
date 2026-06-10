from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ...logging_controller import get_logger
from ..support import ensure_rgba_bgra


logger = get_logger(__name__)


@dataclass
class RembgRemover:
    name: str = "rembg"

    def remove(self, frame_bgr: np.ndarray) -> np.ndarray:
        logger.debug("Running rembg on frame shape=%s", getattr(frame_bgr, "shape", None))
        try:
            from rembg import remove as rembg_remove
        except ImportError as exc:
            raise RuntimeError(
                "rembg is not installed. Install the optional dependency with `pip install -e '.[ai]'`."
            ) from exc

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        result = rembg_remove(pil)

        if isinstance(result, Image.Image):
            rgba_image = result.convert("RGBA")
        elif isinstance(result, bytes):
            rgba_image = Image.open(io.BytesIO(result)).convert("RGBA")
        else:
            raise RuntimeError(f"Unsupported rembg result type: {type(result)!r}")

        rgba = np.array(rgba_image, dtype=np.uint8)
        return ensure_rgba_bgra(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
