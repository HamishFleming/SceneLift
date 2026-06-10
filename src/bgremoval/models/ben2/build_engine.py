from __future__ import annotations

import sys
from pathlib import Path

from ..modnet.build_engine import ModNetEngineConfig, main as modnet_build_main
from ..registry import get_model_spec


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    spec = get_model_spec("ben2-trt")
    onnx_path = Path(spec.metadata["onnx_path"])
    engine_path = Path(spec.metadata["engine_path"])
    args = [
        "--model-key",
        "ben2-trt",
        "--onnx-path",
        str(onnx_path),
        "--engine-path",
        str(engine_path),
        "--input-shape",
        "1,3,1024,1024",
    ]
    if argv:
        args.extend(argv)
    return modnet_build_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
