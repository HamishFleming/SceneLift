from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..logging_controller import get_logger
from .backends.realesrgan import RealESRGANUpscaler
from .base import UpscalerSpec


logger = get_logger(__name__)
_UPSCALER_SPECS: Dict[str, UpscalerSpec] = {}


def register_upscaler(spec: UpscalerSpec) -> UpscalerSpec:
    logger.debug("Registering upscaler spec %s", spec.key)
    _UPSCALER_SPECS[spec.key] = spec
    return spec


def get_upscaler_spec(key: str) -> UpscalerSpec:
    _register_defaults()
    normalized = key.strip().lower()
    try:
        return _UPSCALER_SPECS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown upscaler spec: {key}") from exc


def list_upscaler_specs() -> list[UpscalerSpec]:
    _register_defaults()
    return list(_UPSCALER_SPECS.values())


def _default_weight_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "weights" / "realesrgan"


def _register_defaults() -> None:
    if _UPSCALER_SPECS:
        return

    register_upscaler(
        UpscalerSpec(
            key="realesrgan-x4plus",
            display_name="Real-ESRGAN x4plus",
            kind="realesrgan",
            scale=4,
            local_weights=_default_weight_dir(),
            notes="General-purpose 4x Real-ESRGAN upscaler.",
            metadata={
                "model_filename": "RealESRGAN_x4plus.pth",
                "source_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "num_block": 23,
                "num_feat": 64,
                "num_grow_ch": 32,
            },
        )
    )
    register_upscaler(
        UpscalerSpec(
            key="realesrgan",
            display_name="Real-ESRGAN",
            kind="realesrgan",
            scale=4,
            local_weights=_default_weight_dir(),
            notes="Alias for the default 4x Real-ESRGAN model.",
            metadata={
                "model_filename": "RealESRGAN_x4plus.pth",
                "source_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "num_block": 23,
                "num_feat": 64,
                "num_grow_ch": 32,
            },
        )
    )
    register_upscaler(
        UpscalerSpec(
            key="realesrgan-x4plus-anime",
            display_name="Real-ESRGAN x4plus anime",
            kind="realesrgan",
            scale=4,
            local_weights=_default_weight_dir(),
            notes="Anime-oriented 4x Real-ESRGAN model.",
            metadata={
                "model_filename": "RealESRGAN_x4plus_anime_6B.pth",
                "source_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "num_block": 6,
                "num_feat": 64,
                "num_grow_ch": 32,
            },
        )
    )
    register_upscaler(
        UpscalerSpec(
            key="realesrgan-x2plus",
            display_name="Real-ESRGAN x2plus",
            kind="realesrgan",
            scale=2,
            local_weights=_default_weight_dir(),
            notes="2x Real-ESRGAN model for lighter upscaling.",
            metadata={
                "model_filename": "RealESRGAN_x2plus.pth",
                "num_block": 23,
                "num_feat": 64,
                "num_grow_ch": 32,
            },
        )
    )


def create_upscaler(
    name: str,
    *,
    model_path: str | Path | None = None,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    device: str | None = None,
    half: bool | None = None,
):
    _register_defaults()
    normalized = name.strip().lower()
    logger.info("Selecting upscaler method %s", normalized)
    spec = get_upscaler_spec(normalized)
    resolved_model_path = Path(model_path) if model_path is not None else spec.local_weights / str(spec.metadata["model_filename"])  # type: ignore[union-attr]
    return RealESRGANUpscaler(
        name=spec.key,
        model_path=resolved_model_path,
        scale=int(spec.scale),
        num_feat=int(spec.metadata.get("num_feat", 64)),
        num_block=int(spec.metadata.get("num_block", 23)),
        num_grow_ch=int(spec.metadata.get("num_grow_ch", 32)),
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        device=device,
        half=half,
    )


_register_defaults()
