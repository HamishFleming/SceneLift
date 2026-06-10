from __future__ import annotations

from .models.base import BackgroundRemover
from .models.backends import BiRefNetRemover, GrabCutRemover, RembgRemover, U2NetHumanSegRemover
from .models.registry import create_remover, list_model_specs

__all__ = [
    "BackgroundRemover",
    "BiRefNetRemover",
    "GrabCutRemover",
    "RembgRemover",
    "U2NetHumanSegRemover",
    "available_method_choices",
    "create_remover",
]


def available_method_choices() -> list[str]:
    """Return CLI method names, including shorthand aliases."""
    keys = [spec.key for spec in list_model_specs()]
    return sorted(set(keys + ["modnet", "ben2"]))
