from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..logging_controller import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class HFOnnxDownloadConfig:
    output_path: Path
    repo_id: str
    filename: str
    revision: str = "main"


def download_hf_onnx(config: HFOnnxDownloadConfig) -> Path:
    logger.info("Downloading %s:%s", config.repo_id, config.filename)
    local_cached = _download_hf_file(config.repo_id, config.filename, config.revision)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_cached, config.output_path)
    logger.info("Saved ONNX to %s", config.output_path)
    return config.output_path


def _download_hf_file(repo_id: str, filename: str, revision: str = "main") -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        cache_root = Path.home() / ".cache" / "bgremoval" / "hf"
        local_path = cache_root / repo_id.replace("/", "--") / revision / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            logger.info("Downloading %s from %s", filename, url)
            with urllib.request.urlopen(url) as response, local_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        return local_path

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))


def download_hf_json(repo_id: str, filename: str, revision: str = "main") -> dict[str, Any]:
    path = _download_hf_file(repo_id, filename, revision)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download an ONNX file from Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository id")
    parser.add_argument("--filename", required=True, help="Filename inside the repository")
    parser.add_argument("--output-path", required=True, help="Where to write the ONNX file")
    parser.add_argument("--revision", default="main", help="HF revision or branch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    download_hf_onnx(
        HFOnnxDownloadConfig(
            output_path=Path(args.output_path),
            repo_id=args.repo_id,
            filename=args.filename,
            revision=args.revision,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
