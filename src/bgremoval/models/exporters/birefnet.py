from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ...logging_controller import get_logger
from .common import export_torch_model_to_onnx, load_state_dict_from_checkpoint, resolve_torch_device


logger = get_logger(__name__)


@dataclass(frozen=True)
class BiRefNetExportConfig:
    checkpoint_path: Path
    onnx_path: Path
    model_name: str = "ZhengPeng7/BiRefNet"
    device: str = "cuda"
    input_shape: tuple[int, int, int, int] = (1, 3, 1024, 1024)
    opset: int = 17
    dynamic_axes: bool = False
    checkpoint_key: str | None = None


def _load_birefnet_model(config: BiRefNetExportConfig):
    try:
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "BiRefNet export requires the Hugging Face extras. Install with `pip install -e '.[hf]'`."
        ) from exc

    torch_device = resolve_torch_device(config.device)
    logger.info("Loading BiRefNet model %s", config.model_name)
    model = AutoModelForImageSegmentation.from_pretrained(config.model_name, trust_remote_code=True)
    state_dict = load_state_dict_from_checkpoint(config.checkpoint_path, config.checkpoint_key)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(torch_device)
    model.eval()
    return model, torch_device


def export_to_onnx(config: BiRefNetExportConfig) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for ONNX export.") from exc

    model, device = _load_birefnet_model(config)
    dummy = torch.randn(*config.input_shape, device=device)

    class _BiRefNetOnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            outputs = self.model(inputs)
            if isinstance(outputs, (list, tuple)):
                return outputs[-1]
            if hasattr(outputs, "logits"):
                return outputs.logits
            return outputs

    dynamic_axes = None
    if config.dynamic_axes:
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "mask": {0: "batch", 2: "height", 3: "width"},
        }

    wrapped = _BiRefNetOnnxWrapper(model)
    return export_torch_model_to_onnx(
        wrapped,
        dummy,
        config.onnx_path,
        input_names=["image"],
        output_names=["mask"],
        opset_version=config.opset,
        dynamic_axes=dynamic_axes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export BiRefNet to ONNX")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the BiRefNet .pth checkpoint")
    parser.add_argument("--onnx-path", required=True, help="Output ONNX file")
    parser.add_argument(
        "--model-name",
        default="ZhengPeng7/BiRefNet",
        help="Hugging Face model name or local config directory that defines the BiRefNet architecture",
    )
    parser.add_argument("--device", default="cuda", help="Export device, usually cuda or cpu")
    parser.add_argument("--input-shape", default="1,3,1024,1024", type=_parse_shape, help="Input shape")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic-axes", action="store_true", help="Enable dynamic axes in ONNX export")
    parser.add_argument("--checkpoint-key", default=None, help="Optional dict key to extract from checkpoint")
    return parser


def _parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("input shape must have four comma-separated integers")
    return tuple(parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_to_onnx(
        BiRefNetExportConfig(
            checkpoint_path=Path(args.checkpoint_path),
            onnx_path=Path(args.onnx_path),
            model_name=args.model_name,
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
