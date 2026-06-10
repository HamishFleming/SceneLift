from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...logging_controller import get_logger
from ..base import BackgroundRemover
from ..tensorrt.session import TensorRTSession, load_engine


logger = get_logger(__name__)


@dataclass
class ModNetTensorRTRemover:
    name: str = "modnet-trt"
    engine_path: Path = Path(__file__).resolve().parents[1] / "weights" / "modnet" / "modnet.engine"
    input_size: tuple[int, int] = (512, 512)
    session: TensorRTSession | None = None

    def _get_session(self) -> TensorRTSession:
        if self.session is None:
            logger.info("Loading MODNet TensorRT engine from %s", self.engine_path)
            self.session = load_engine(
                self.engine_path,
                input_shapes={"input": (1, 3, self.input_size[1], self.input_size[0])},
            )
        return self.session

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, self.input_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)

    def _postprocess(self, frame_bgr: np.ndarray, matte: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, self.input_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = rgb * matte[..., None]
        return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGRA)

    def remove(self, frame_bgr: np.ndarray) -> np.ndarray:
        session = self._get_session()
        inp = self._preprocess(frame_bgr)
        out = session.infer(inp.ravel())
        matte = out.reshape(self.input_size[1], self.input_size[0])
        matte = np.clip(matte, 0.0, 1.0)
        return self._postprocess(frame_bgr, matte)

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
