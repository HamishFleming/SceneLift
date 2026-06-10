from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from ...logging_controller import get_logger


logger = get_logger(__name__)


@dataclass
class GrabCutRemover:
    name: str = "grabcut"
    iterations: int = 5

    def remove(self, frame_bgr: np.ndarray) -> np.ndarray:
        logger.debug("Running GrabCut on frame shape=%s", getattr(frame_bgr, "shape", None))
        frame_bgr = np.asarray(frame_bgr)
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("GrabCutRemover expects a BGR frame")

        h, w = frame_bgr.shape[:2]
        if h < 10 or w < 10:
            rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255
            return rgba

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (
            int(w * 0.1),
            int(h * 0.1),
            max(1, int(w * 0.8)),
            max(1, int(h * 0.8)),
        )

        try:
            cv2.grabCut(frame_bgr, mask, rect, bgd_model, fgd_model, self.iterations, cv2.GC_INIT_WITH_RECT)
            alpha = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
            alpha = cv2.medianBlur(alpha, 5)
        except cv2.error:
            logger.exception("GrabCut failed; falling back to opaque alpha")
            alpha = np.full((h, w), 255, dtype=np.uint8)

        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        return rgba
