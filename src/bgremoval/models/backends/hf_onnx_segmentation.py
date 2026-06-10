from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ...logging_controller import get_logger
from ..hf_onnx import _download_hf_file


logger = get_logger(__name__)


@dataclass
class HFOnnxSegmentationRemover:
    name: str
    model_name: str
    onnx_filename: str
    processor_filename: str
    revision: str = "main"
    device: str | None = None
    weights_path: str | None = None

    _session: Any = None
    _processor_config: dict[str, Any] | None = None
    _input_name: str | None = None
    _output_name: str | None = None
    _providers: list[str] | None = None

    @property
    def _local_root(self) -> Path:
        if self.weights_path:
            return Path(self.weights_path)
        slug = self.name.replace("/", "-")
        return Path(__file__).resolve().parents[1] / "weights" / slug

    def _ensure_assets(self) -> tuple[Path, Path]:
        root = self._local_root
        onnx_path = root / self.onnx_filename
        processor_path = root / self.processor_filename

        if onnx_path.exists() and processor_path.exists():
            return onnx_path, processor_path

        logger.info("Downloading %s assets from Hugging Face", self.name)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        processor_path.parent.mkdir(parents=True, exist_ok=True)

        cached_onnx = _download_hf_file(self.model_name, self.onnx_filename, self.revision)
        cached_processor = _download_hf_file(self.model_name, self.processor_filename, self.revision)
        shutil.copy2(cached_onnx, onnx_path)
        shutil.copy2(cached_processor, processor_path)

        if not onnx_path.exists() or not processor_path.exists():
            raise RuntimeError(f"Failed to download {self.name} assets from Hugging Face")

        return onnx_path, processor_path

    def _load_processor_config(self) -> dict[str, Any]:
        if self._processor_config is not None:
            return self._processor_config
        _, processor_path = self._ensure_assets()
        with processor_path.open("r", encoding="utf-8") as handle:
            self._processor_config = json.load(handle)
        return self._processor_config

    def _load_session(self):
        if self._session is not None:
            return self._session

        onnx_path, _ = self._ensure_assets()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                f"{self.name} requires onnxruntime. Install with `pip install -e '.[hf]'`."
            ) from exc

        available = set(ort.get_available_providers())
        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "OpenVINOExecutionProvider" in available:
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        logger.info("Loading %s ONNX session from %s with providers=%s", self.name, onnx_path, providers)
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._providers = providers
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        self._session = session
        return session

    def _resize_image(self, image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
        size = cfg.get("size") or {}
        width = size.get("width")
        height = size.get("height")
        longest_edge = size.get("longest_edge")
        keep_aspect = bool(cfg.get("keep_aspect_ratio", False))

        if width and height:
            return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)

        if longest_edge:
            longest_edge = int(longest_edge)
            h, w = image.shape[:2]
            if max(h, w) == longest_edge:
                return image
            scale = longest_edge / float(max(h, w))
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        if keep_aspect and width and not height:
            h, w = image.shape[:2]
            scale = int(width) / float(max(h, w))
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        return image

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
        cfg = self._load_processor_config()
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if cfg.get("do_convert_rgb", True) else frame_bgr
        pil = Image.fromarray(image).convert("RGB")
        original_size = pil.size
        arr = np.asarray(pil, dtype=np.float32)
        arr = self._resize_image(arr, cfg) if cfg.get("do_resize", True) else arr

        if cfg.get("do_rescale", True):
            rescale_factor = float(cfg.get("rescale_factor", 1.0 / 255.0))
            arr = arr * rescale_factor

        if cfg.get("do_normalize", False):
            mean = np.array(cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std = np.array(cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
            arr = (arr - mean) / std

        arr = np.transpose(arr, (2, 0, 1))
        c, h, w = arr.shape

        pad_top = pad_bottom = pad_left = pad_right = 0
        if cfg.get("do_pad", False):
            pad_cfg = cfg.get("pad_size", {}) or {}
            target_w = int(pad_cfg.get("width", w))
            target_h = int(pad_cfg.get("height", h))
            pad_top = max(0, (target_h - h) // 2)
            pad_bottom = max(0, target_h - h - pad_top)
            pad_left = max(0, (target_w - w) // 2)
            pad_right = max(0, target_w - w - pad_left)
            if pad_top or pad_bottom or pad_left or pad_right:
                arr = np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

        tensor = arr.astype(np.float32, copy=False)[None, ...]
        logger.debug(
            "Prepared %s input original=%s resized=%s padded=%s",
            self.name,
            original_size,
            (w, h),
            (pad_left + w + pad_right, pad_top + h + pad_bottom),
        )
        return tensor, original_size, (pad_top, pad_bottom, pad_left, pad_right)

    def _extract_mask(self, outputs: Any) -> np.ndarray:
        if isinstance(outputs, dict):
            outputs = next(iter(outputs.values()))
        elif isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        mask = np.asarray(outputs)
        mask = np.squeeze(mask)
        if mask.ndim == 3 and mask.shape[0] in {1, 3}:
            mask = mask[0]
        if mask.ndim != 2:
            raise RuntimeError(f"Unexpected {self.name} output shape: {mask.shape!r}")
        return mask

    def _to_alpha(self, mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        mask = mask.astype(np.float32, copy=False)
        if mask.size and (mask.min() < 0.0 or mask.max() > 1.5):
            if mask.min() >= -20.0 and mask.max() <= 20.0:
                mask = 1.0 / (1.0 + np.exp(-mask))
            elif mask.max() > 255.0:
                mask = np.clip(mask, 0.0, 1.0)

        if mask.max() <= 1.5:
            mask = mask * 255.0

        alpha = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0, 255).astype(np.uint8)

    def remove(self, frame_bgr: np.ndarray) -> np.ndarray:
        session = self._load_session()
        tensor, original_size, pads = self._preprocess(frame_bgr)
        input_name = self._input_name or session.get_inputs()[0].name
        output_name = self._output_name or session.get_outputs()[0].name

        logger.debug("Running %s inference on frame shape=%s", self.name, getattr(frame_bgr, "shape", None))
        outputs = session.run([output_name], {input_name: tensor})
        mask = self._extract_mask(outputs)

        pad_top, pad_bottom, pad_left, pad_right = pads
        if pad_top or pad_bottom or pad_left or pad_right:
            y2 = mask.shape[0] - pad_bottom if pad_bottom else mask.shape[0]
            x2 = mask.shape[1] - pad_right if pad_right else mask.shape[1]
            mask = mask[pad_top:y2, pad_left:x2]

        alpha = self._to_alpha(mask, original_size)
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        logger.debug("%s inference completed", self.name)
        return rgba
