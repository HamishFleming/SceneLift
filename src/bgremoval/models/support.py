from __future__ import annotations

import cv2
import numpy as np


def ensure_rgba_bgra(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError("Expected a 3D image array")
    if array.shape[2] == 4:
        return array
    if array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2BGRA)
    raise ValueError(f"Unsupported channel count: {array.shape[2]}")
