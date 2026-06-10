"""Background removal prototype package."""

from .api import run
from .live import LiveConfig, run_live_virtualcam
from .logging_controller import get_logger, setup_logging
from .models import ModelSpec, create_remover, get_model_spec, list_model_specs, register_model

__all__ = [
    "run",
    "LiveConfig",
    "run_live_virtualcam",
    "setup_logging",
    "get_logger",
    "ModelSpec",
    "create_remover",
    "get_model_spec",
    "list_model_specs",
    "register_model",
]
