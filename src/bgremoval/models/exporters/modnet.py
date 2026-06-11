from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ...logging_controller import get_logger
from ..modnet.official import OfficialModNetSpec, build_official_modnet_model
from .common import export_torch_model_to_onnx


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModNetExportConfig:
    repo_root: Path
    checkpoint_path: Path
    onnx_path: Path
    device: str = "cuda"
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512)
    opset: int = 17
    dynamic_axes: bool = False
    checkpoint_key: str | None = None


def _parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("input shape must have four comma-separated integers")
    return tuple(parts)  # type: ignore[return-value]


def export_to_onnx(config: ModNetExportConfig) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for ONNX export.") from exc

    model = build_official_modnet_model(
        OfficialModNetSpec(
            repo_root=config.repo_root,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            checkpoint_key=config.checkpoint_key,
        )
    )
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model = model.to(device)
    dummy = torch.randn(*config.input_shape, device=device)

    dynamic_axes = None
    if config.dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    export_model = model.module if hasattr(model, "module") else model
    return export_torch_model_to_onnx(
        export_model,
        dummy,
        config.onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=config.opset,
        dynamic_axes=dynamic_axes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MODNet to ONNX")
    parser.add_argument("--repo-root", required=True, help="Path to the official MODNet repository")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the MODNet checkpoint")
    parser.add_argument("--onnx-path", required=True, help="Output ONNX file")
    parser.add_argument("--device", default="cuda", help="Export device, usually cuda or cpu")
    parser.add_argument("--input-shape", default="1,3,512,512", type=_parse_shape, help="Input shape")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic-axes", action="store_true", help="Enable dynamic axes in ONNX export")
    parser.add_argument("--checkpoint-key", default=None, help="Optional dict key to extract from checkpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_to_onnx(
        ModNetExportConfig(
            repo_root=Path(args.repo_root),
            checkpoint_path=Path(args.checkpoint_path),
            onnx_path=Path(args.onnx_path),
            device=args.device,
            input_shape=args.input_shape,
            opset=args.opset,
            dynamic_axes=args.dynamic_axes,
            checkpoint_key=args.checkpoint_key,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
