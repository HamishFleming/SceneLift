from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ...logging_controller import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class OfficialModNetSpec:
    repo_root: Path
    checkpoint_path: Path
    device: str = "cuda"
    checkpoint_key: str | None = None


@contextmanager
def _prep_repo_root(repo_root: Path):
    repo_root = repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(repo_root))
        except ValueError:
            pass


def load_official_modnet_module(repo_root: Path) -> ModuleType:
    with _prep_repo_root(repo_root):
        logger.debug("Importing official MODNet module from %s", repo_root)
        return importlib.import_module("onnx.modnet_onnx")


def build_official_modnet_model(spec: OfficialModNetSpec):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Torch is required to load the MODNet export model.") from exc

    module = load_official_modnet_module(spec.repo_root)
    if not hasattr(module, "MODNet"):
        raise RuntimeError("The official MODNet module does not expose a MODNet class")

    modnet = module.MODNet(backbone_pretrained=False)
    device = torch.device(spec.device if torch.cuda.is_available() or spec.device == "cpu" else "cpu")
    logger.info("Loading MODNet checkpoint %s", spec.checkpoint_path)
    state = torch.load(spec.checkpoint_path, map_location=device)
    if spec.checkpoint_key is not None and isinstance(state, dict) and spec.checkpoint_key in state:
        state = state[spec.checkpoint_key]
    if isinstance(state, dict) and "state_dict" in state and not any(k.startswith("module.") for k in state):
        state = state["state_dict"]

    modnet.load_state_dict(state)
    modnet = modnet.to(device)
    modnet.eval()
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        modnet = nn.DataParallel(modnet)
    return modnet
