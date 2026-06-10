from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ...logging_controller import get_logger
from ..hf_onnx import HFOnnxDownloadConfig, download_hf_onnx


logger = get_logger(__name__)

XENOVA_REPO_ID = "Xenova/modnet"
DEFAULT_ONNX_FILENAME = "onnx/model.onnx"
SUPPORTED_ONNX_VARIANTS = {
    "fp32": "onnx/model.onnx",
    "fp16": "onnx/model_fp16.onnx",
    "quantized": "onnx/model_quantized.onnx",
    "uint8": "onnx/model_uint8.onnx",
    "q4": "onnx/model_q4.onnx",
    "q4f16": "onnx/model_q4f16.onnx",
    "bnb4": "onnx/model_bnb4.onnx",
}


@dataclass(frozen=True)
class XenovaModNetConfig:
    output_path: Path
    variant: str = "fp32"
    revision: str = "main"
    repo_id: str = XENOVA_REPO_ID


def download_xenova_modnet_onnx(config: XenovaModNetConfig) -> Path:
    filename = SUPPORTED_ONNX_VARIANTS.get(config.variant)
    if filename is None:
        raise ValueError(f"Unsupported Xenova MODNet variant: {config.variant}")
    return download_hf_onnx(
        HFOnnxDownloadConfig(
            output_path=config.output_path,
            repo_id=config.repo_id,
            filename=filename,
            revision=config.revision,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Xenova/modnet ONNX weights")
    parser.add_argument("--output-path", required=True, help="Where to write the ONNX file")
    parser.add_argument(
        "--variant",
        default="fp32",
        choices=sorted(SUPPORTED_ONNX_VARIANTS.keys()),
        help="ONNX variant to download",
    )
    parser.add_argument("--revision", default="main", help="HF revision or branch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    download_xenova_modnet_onnx(
        XenovaModNetConfig(
            output_path=Path(args.output_path),
            variant=args.variant,
            revision=args.revision,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
