from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType
from pathlib import Path

import cv2
import numpy as np

from ...logging_controller import get_logger


logger = get_logger(__name__)


def _ensure_torchvision_compat() -> None:
    try:
        import torchvision.transforms.functional as functional
    except ImportError:
        return

    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ImportError:
        module = ModuleType("torchvision.transforms.functional_tensor")
        module.rgb_to_grayscale = functional.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = module


@dataclass
class RealESRGANUpscaler:
    name: str = "realesrgan"
    model_path: Path | None = None
    scale: int = 4
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 0
    device: str | None = None
    half: bool | None = None

    _upsampler: object | None = None

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Real-ESRGAN upscaling requires torch. Install the model dependencies first.") from exc
        return torch

    def _load(self):
        if self._upsampler is not None:
            return self._upsampler

        if self.model_path is None:
            raise RuntimeError(
                "Real-ESRGAN requires a model checkpoint path. Pass --model-path or place the weights under "
                "src/bgremoval/models/weights/realesrgan/"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"Real-ESRGAN model checkpoint does not exist: {self.model_path}")

        try:
            _ensure_torchvision_compat()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise RuntimeError(
                "Real-ESRGAN upscaling requires the `realesrgan`, `basicsr`, `torch`, and `torchvision` packages. "
                "Install them with `pip install -e '.[upscale]'` before using the upscaler commands."
            ) from exc

        torch = self._torch()
        if self.device:
            device = torch.device(self.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        half = self.half if self.half is not None else device.type == "cuda"

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=self.num_feat,
            num_block=self.num_block,
            num_grow_ch=self.num_grow_ch,
            scale=self.scale,
        )
        logger.info("Loading Real-ESRGAN checkpoint %s", self.model_path)
        self._upsampler = RealESRGANer(
            scale=self.scale,
            model_path=str(self.model_path),
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=half,
            gpu_id=None if device.type == "cpu" else (device.index or 0),
        )
        return self._upsampler

    def upscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        upsampler = self._load()
        output, _ = upsampler.enhance(frame_bgr, outscale=self.scale)
        if output is None:
            raise RuntimeError("Real-ESRGAN returned no output frame")
        if output.ndim != 3:
            raise RuntimeError(f"Unexpected Real-ESRGAN output shape: {getattr(output, 'shape', None)}")
        return output

    def close(self) -> None:
        self._upsampler = None
