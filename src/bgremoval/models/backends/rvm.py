from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ...logging_controller import get_logger


logger = get_logger(__name__)


@dataclass
class RVMRemover:
    name: str = "rvm"
    model_variant: str = "mobilenetv3"
    repo: str = "PeterL1n/RobustVideoMatting"
    device: str | None = None
    downsample_ratio: float = 0.25

    _model: Any = None
    _torch_device: Any = None
    _rec: list[Any] | None = None
    _last_frame_size: tuple[int, int] | None = None

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("rvm requires torch. Install it before using the RVM backend.") from exc
        return torch

    def _resolve_device(self):
        torch = self._torch()
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        torch = self._torch()
        logger.info("Loading RVM model %s from %s", self.model_variant, self.repo)
        try:
            model = torch.hub.load(self.repo, self.model_variant, trust_repo=True)
        except TypeError:
            # Older torch versions do not support trust_repo.
            model = torch.hub.load(self.repo, self.model_variant)
        device = self._resolve_device()
        model = model.to(device)
        model.eval()
        if device.type == "cuda":
            model = model.half()
        self._torch_device = device
        self._rec = [None] * 4
        return model

    def _reset_state(self) -> None:
        self._rec = [None] * 4
        self._last_frame_size = None

    def _preprocess(self, frame_bgr: np.ndarray):
        torch = self._torch()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        if self._torch_device is None:
            self._torch_device = self._resolve_device()
        tensor = tensor.to(self._torch_device)
        if self._torch_device.type == "cuda":
            tensor = tensor.half()
        return tensor

    def remove(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self._torch()
        if self._model is None:
            self._model = self._load_model()

        frame_size = tuple(int(v) for v in frame_bgr.shape[:2])
        if self._last_frame_size is not None and self._last_frame_size != frame_size:
            logger.info("Resetting RVM recurrent state due to frame-size change %s -> %s", self._last_frame_size, frame_size)
            self._reset_state()
        self._last_frame_size = frame_size
        if self._rec is None:
            self._rec = [None] * 4

        src = self._preprocess(frame_bgr)
        with torch.no_grad():
            fgr, pha, *self._rec = self._model(src, *self._rec, self.downsample_ratio)
            fgr = fgr[0].to(torch.float32).clamp(0, 1)
            pha = pha[0, 0].to(torch.float32).clamp(0, 1)

        foreground = (fgr.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        alpha = (pha.cpu().numpy() * 255.0).astype(np.uint8)
        bgra = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGRA)
        bgra[:, :, 3] = alpha
        return bgra

    def close(self) -> None:
        self._reset_state()
        self._model = None
        self._torch_device = None
