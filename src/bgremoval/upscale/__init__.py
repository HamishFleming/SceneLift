from __future__ import annotations

from .base import ImageUpscaler
from .registry import create_upscaler, get_upscaler_spec, list_upscaler_specs, register_upscaler

__all__ = [
    "ImageUpscaler",
    "create_upscaler",
    "get_upscaler_spec",
    "list_upscaler_specs",
    "register_upscaler",
]
