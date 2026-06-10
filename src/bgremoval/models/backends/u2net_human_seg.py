from __future__ import annotations

from dataclasses import dataclass

from .hf_onnx_segmentation import HFOnnxSegmentationRemover


@dataclass
class U2NetHumanSegRemover(HFOnnxSegmentationRemover):
    name: str = "u2net-human-seg"
    model_name: str = "BritishWerewolf/U-2-Net-Human-Seg"
    onnx_filename: str = "onnx/model.onnx"
    processor_filename: str = "preprocessor_config.json"

