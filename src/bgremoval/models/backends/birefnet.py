from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ...logging_controller import get_logger


logger = get_logger(__name__)


@dataclass
class BiRefNetRemover:
    name: str = "birefnet"
    model_name: str = "ZhengPeng7/BiRefNet"
    device: str | None = None
    input_size: tuple[int, int] = (1024, 1024)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    weights_path: str | None = None

    _model: Any = None
    _torch_device: Any = None

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("birefnet requires torch. Install it before using the BiRefNet backend.") from exc
        return torch

    def _resolve_device(self):
        torch = self._torch()
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        logger.info("Loading BiRefNet model %s", self.model_name)
        try:
            from transformers import AutoModelForImageSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "birefnet requires the Hugging Face extras. Install with `pip install -e '.[hf]'`."
            ) from exc

        if self.weights_path:
            logger.info("BiRefNet weights path configured: %s", self.weights_path)

        try:
            model = AutoModelForImageSegmentation.from_pretrained(self.model_name, trust_remote_code=True)
        except ImportError as exc:
            raise RuntimeError(
                "BiRefNet remote code is missing a dependency. Install the HF extra with "
                "`pip install -e '.[hf]'` to include einops, kornia, timm, and torchvision."
            ) from exc
        device = self._resolve_device()
        model = model.to(device)
        model.eval()
        if device.type == "cuda":
            model = model.half()
        self._torch_device = device
        return model

    def _preprocess(self, frame_bgr: np.ndarray):
        torch = self._torch()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).convert("RGB")
        original_size = pil.size
        resized = pil.resize(self.input_size, Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        if self._torch_device is None:
            self._torch_device = self._resolve_device()
        tensor = tensor.to(self._torch_device)
        if self._torch_device.type == "cuda":
            tensor = tensor.half()
        return tensor, original_size

    def remove(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self._torch()
        if self._model is None:
            self._model = self._load_model()

        inputs, original_size = self._preprocess(frame_bgr)
        logger.debug("Running BiRefNet inference on size=%s", original_size)
        with torch.no_grad():
            outputs = self._model(inputs)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[-1]
            elif hasattr(outputs, "logits"):
                pred = outputs.logits
            else:
                pred = outputs
            pred = pred.sigmoid().to(torch.float32)

        if pred.ndim == 4:
            pred = pred[0]
        if pred.ndim == 3:
            pred = pred[:1]
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        if pred.ndim != 3:
            raise RuntimeError(f"Unexpected BiRefNet output shape: {tuple(pred.shape)}")

        mask = torch.nn.functional.interpolate(
            pred.unsqueeze(0),
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        alpha = (mask.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        logger.debug("BiRefNet inference completed")
        return rgba
