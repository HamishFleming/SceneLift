from __future__ import annotations

from pathlib import Path
from typing import Any

from ...logging_controller import get_logger


logger = get_logger(__name__)
_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights")
_STRIP_PREFIXES = ("module.", "model.", "net.")


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - import guarded at runtime
        raise RuntimeError("Torch is required for ONNX export.") from exc
    return torch


def resolve_torch_device(requested: str):
    torch = _torch()
    normalized = requested.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        logger.warning("CUDA was requested for ONNX export, but no GPU is available. Falling back to CPU.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _unwrap_checkpoint(state: Any, checkpoint_key: str | None) -> Any:
    if checkpoint_key and isinstance(state, dict) and checkpoint_key in state:
        return state[checkpoint_key]
    if isinstance(state, dict):
        for key in _STATE_DICT_KEYS:
            candidate = state.get(key)
            if isinstance(candidate, dict):
                return candidate
    return state


def _strip_known_prefixes(key: str) -> str:
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
    return stripped


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = _strip_known_prefixes(key)
        if new_key in normalized and new_key != key:
            raise RuntimeError(f"Duplicate checkpoint key after normalization: {new_key}")
        normalized[new_key] = value
    return normalized


def load_state_dict_from_checkpoint(checkpoint_path: Path, checkpoint_key: str | None = None) -> dict[str, Any]:
    torch = _torch()
    logger.info("Loading checkpoint %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    state = _unwrap_checkpoint(state, checkpoint_key)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} did not contain a state dict")
    return normalize_state_dict(state)


def export_torch_model_to_onnx(
    model,
    dummy_input,
    onnx_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    opset_version: int,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    torch = _torch()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting ONNX to %s", onnx_path)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    logger.info("Export complete: %s", onnx_path)
    return onnx_path
