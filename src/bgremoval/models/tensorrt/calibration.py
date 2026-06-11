from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ...logging_controller import get_logger


logger = get_logger(__name__)


def _require_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT is not installed. Install the 'trt' extra first.") from exc
    return trt


def _require_pycuda():
    try:
        import pycuda.driver as cuda
    except ImportError as exc:
        raise RuntimeError("pycuda is not installed. Install the 'trt' extra first.") from exc
    return cuda


def collect_image_paths(
    image_dir: Path,
    *,
    max_samples: int,
    globs: Iterable[str] = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"),
) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Calibration directory does not exist: {image_dir}")
    paths: list[Path] = []
    for pattern in globs:
        paths.extend(sorted(image_dir.rglob(pattern)))
    unique_paths = sorted({path.resolve() for path in paths if path.is_file()})
    if max_samples > 0:
        unique_paths = unique_paths[:max_samples]
    return unique_paths


def build_calibration_batches(
    image_paths: Iterable[Path],
    *,
    batch_size: int,
    preprocess: Callable[[Path], np.ndarray],
) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for path in image_paths:
        sample = preprocess(path)
        if sample.ndim != 3:
            raise RuntimeError(f"Calibration sample {path} must be CHW; got shape {sample.shape}")
        current.append(np.ascontiguousarray(sample, dtype=np.float32))
        if len(current) == batch_size:
            batches.append(np.stack(current, axis=0))
            current = []
    return batches


def create_image_folder_int8_calibrator(
    batches: list[np.ndarray],
    *,
    batch_size: int,
    cache_path: Path | None = None,
):
    trt = _require_tensorrt()
    cuda = _require_pycuda()

    class _ImageFolderInt8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.batches = [np.ascontiguousarray(batch, dtype=np.float32) for batch in batches]
            self.batch_size = batch_size
            self.cache_path = cache_path
            self.current_index = 0
            self.device_input = None
            if self.batches:
                self.device_input = cuda.mem_alloc(self.batches[0].nbytes)
            elif self.cache_path is None or not self.cache_path.exists():
                raise RuntimeError("INT8 calibration requires at least one full batch of samples or an existing cache")

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names):  # noqa: D401 - TensorRT calls this signature
            if self.current_index >= len(self.batches):
                return None
            if self.device_input is None:
                return None
            batch = self.batches[self.current_index]
            self.current_index += 1
            cuda.memcpy_htod(self.device_input, batch.ravel())
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path is None or not self.cache_path.exists():
                return None
            logger.info("Loaded TensorRT INT8 calibration cache from %s", self.cache_path)
            return self.cache_path.read_bytes()

        def write_calibration_cache(self, cache):
            if self.cache_path is None:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(cache)
            logger.info("Saved TensorRT INT8 calibration cache to %s", self.cache_path)

    return _ImageFolderInt8Calibrator()
