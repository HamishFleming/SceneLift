from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ...logging_controller import get_logger
from ..hf_onnx import HFOnnxDownloadConfig, download_hf_onnx
from ..registry import get_model_spec


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download BEN2 ONNX weights")
    parser.add_argument("--output-path", default=None, help="Optional override for the ONNX file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    spec = get_model_spec("ben2-trt")
    output_path = Path(args.output_path) if args.output_path else Path(spec.metadata["onnx_path"])
    download_hf_onnx(
        HFOnnxDownloadConfig(
            output_path=output_path,
            repo_id=str(spec.metadata["source_repo"]),
            filename=str(spec.metadata["source_onnx"]),
            revision=str(spec.metadata.get("source_revision", "main")),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
