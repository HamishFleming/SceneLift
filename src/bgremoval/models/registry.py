from __future__ import annotations

from pathlib import Path
from typing import Dict

from .backends.birefnet import BiRefNetRemover
from .backends.grabcut import GrabCutRemover
from .backends.mediapipe_selfie_segmentation import MediaPipeSelfieSegmentationRemover
from .backends.rembg_backend import RembgRemover
from .backends.u2net_human_seg import U2NetHumanSegRemover
from .base import ModelSpec
from .modnet.trt_backend import ModNetTensorRTRemover
from ..logging_controller import get_logger


logger = get_logger(__name__)
_MODEL_SPECS: Dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    logger.debug("Registering model spec %s", spec.key)
    _MODEL_SPECS[spec.key] = spec
    return spec


def get_model_spec(key: str) -> ModelSpec:
    _register_defaults()
    normalized = key.strip().lower()
    try:
        return _MODEL_SPECS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown model spec: {key}") from exc


def list_model_specs() -> list[ModelSpec]:
    _register_defaults()
    return list(_MODEL_SPECS.values())


def _default_weight_dir() -> Path:
    return Path(__file__).resolve().parent / "weights"


def _register_defaults() -> None:
    if _MODEL_SPECS:
        return

    register_model(
        ModelSpec(
            key="grabcut",
            display_name="GrabCut baseline",
            kind="classical",
            supports_video=True,
            notes="No external weights required.",
        )
    )
    register_model(
        ModelSpec(
            key="rembg",
            display_name="rembg",
            kind="ai",
            supports_video=True,
            notes="Requires rembg and an available backend model.",
        )
    )
    register_model(
        ModelSpec(
            key="birefnet",
            display_name="BiRefNet",
            kind="ai",
            huggingface_id="ZhengPeng7/BiRefNet",
            local_weights=_default_weight_dir() / "birefnet",
            supports_video=True,
            notes="Official Hugging Face weights for the BiRefNet segmentation model.",
        )
    )
    register_model(
        ModelSpec(
            key="u2net-human-seg",
            display_name="U-2-Net Human Seg",
            kind="ai",
            huggingface_id="BritishWerewolf/U-2-Net-Human-Seg",
            local_weights=_default_weight_dir() / "u2net-human-seg",
            supports_video=True,
            notes="Human segmentation model from Hugging Face for lightweight background removal.",
            metadata={
                "source_repo": "BritishWerewolf/U-2-Net-Human-Seg",
                "source_onnx": "onnx/model.onnx",
                "source_processor": "preprocessor_config.json",
                "source_revision": "main",
                "onnx_path": str(_default_weight_dir() / "u2net-human-seg" / "onnx" / "model.onnx"),
                "processor_path": str(_default_weight_dir() / "u2net-human-seg" / "preprocessor_config.json"),
            },
        )
    )
    register_model(
        ModelSpec(
            key="mediapipe-selfie-segmentation",
            display_name="MediaPipe Selfie Segmentation",
            kind="ai",
            huggingface_id="onnx-community/mediapipe_selfie_segmentation",
            local_weights=_default_weight_dir() / "mediapipe-selfie-segmentation",
            supports_video=True,
            notes="Fast ONNX-backed selfie segmentation model from the ONNX Community.",
            metadata={
                "source_repo": "onnx-community/mediapipe_selfie_segmentation",
                "source_onnx": "onnx/model.onnx",
                "source_processor": "preprocessor_config.json",
                "source_revision": "main",
                "onnx_path": str(_default_weight_dir() / "mediapipe-selfie-segmentation" / "onnx" / "model.onnx"),
                "processor_path": str(_default_weight_dir() / "mediapipe-selfie-segmentation" / "preprocessor_config.json"),
            },
        )
    )
    register_model(
        ModelSpec(
            key="modnet-trt",
            display_name="MODNet TensorRT",
            kind="tensorrt",
            local_weights=_default_weight_dir() / "modnet",
            supports_video=True,
            notes="Xenova/modnet ONNX source, TensorRT engine, and runtime scaffold for MODNet-style live background removal.",
            metadata={
                "source_repo": "Xenova/modnet",
                "source_onnx": "onnx/model.onnx",
                "source_variant": "fp32",
                "onnx_path": str(_default_weight_dir() / "modnet" / "onnx" / "model.onnx"),
                "engine_path": str(_default_weight_dir() / "modnet" / "modnet.engine"),
            },
        )
    )
    register_model(
        ModelSpec(
            key="ben2-trt",
            display_name="BEN2 ONNX/TensorRT",
            kind="tensorrt",
            local_weights=_default_weight_dir() / "ben2",
            supports_video=True,
            notes="PramaLLC/BEN2 source, TensorRT engine, and live background-removal scaffold.",
            metadata={
                "source_repo": "PramaLLC/BEN2",
                "source_onnx": "BEN2_Base.onnx",
                "source_variant": "fp16",
                "onnx_path": str(_default_weight_dir() / "ben2" / "onnx" / "BEN2_Base.onnx"),
                "engine_path": str(_default_weight_dir() / "ben2" / "ben2.engine"),
            },
        )
    )


def create_remover(name: str):
    _register_defaults()
    normalized = name.strip().lower()
    logger.info("Selecting remover method %s", normalized)
    if normalized == "grabcut":
        return GrabCutRemover()
    if normalized == "rembg":
        return RembgRemover()
    if normalized == "birefnet":
        spec = get_model_spec("birefnet")
        return BiRefNetRemover(model_name=spec.huggingface_id or "ZhengPeng7/BiRefNet")
    if normalized == "u2net-human-seg":
        spec = get_model_spec("u2net-human-seg")
        return U2NetHumanSegRemover(
            model_name=spec.huggingface_id or "BritishWerewolf/U-2-Net-Human-Seg",
            revision=str(spec.metadata.get("source_revision", "main")),
            weights_path=str(spec.local_weights) if spec.local_weights else None,
        )
    if normalized == "mediapipe-selfie-segmentation":
        spec = get_model_spec("mediapipe-selfie-segmentation")
        return MediaPipeSelfieSegmentationRemover(
            model_name=spec.huggingface_id or "onnx-community/mediapipe_selfie_segmentation",
            revision=str(spec.metadata.get("source_revision", "main")),
            weights_path=str(spec.local_weights) if spec.local_weights else None,
        )
    if normalized == "modnet-trt":
        spec = get_model_spec("modnet-trt")
        engine_path = Path(spec.metadata["engine_path"])
        return ModNetTensorRTRemover(engine_path=engine_path)
    if normalized == "ben2-trt":
        spec = get_model_spec("ben2-trt")
        engine_path = Path(spec.metadata["engine_path"])
        return ModNetTensorRTRemover(engine_path=engine_path, input_size=(1024, 1024), name="ben2-trt")
    raise ValueError(f"Unknown remover method: {name}")


_register_defaults()
