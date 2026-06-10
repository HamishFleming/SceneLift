from __future__ import annotations

from dataclasses import dataclass

from .hf_onnx_segmentation import HFOnnxSegmentationRemover


@dataclass
class MediaPipeSelfieSegmentationRemover(HFOnnxSegmentationRemover):
    name: str = "mediapipe-selfie-segmentation"
    model_name: str = "onnx-community/mediapipe_selfie_segmentation"
    onnx_filename: str = "onnx/model.onnx"
    processor_filename: str = "preprocessor_config.json"

