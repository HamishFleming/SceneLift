from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import urllib.request

from ..logging_controller import get_logger, setup_logging
from .hf_onnx import HFOnnxDownloadConfig, _download_hf_file, download_hf_onnx
from .registry import get_model_spec, list_model_specs
from ..upscale.registry import list_upscaler_specs


logger = get_logger(__name__)


@dataclass(frozen=True)
class PullResult:
    key: str
    status: str
    files: list[str]
    skipped_reason: str | None = None
    error: str | None = None


def _default_weights_root() -> Path:
    return Path(__file__).resolve().parent / "weights"


def _relocate_path(source_path: Path, weights_root: Path) -> Path:
    default_root = _default_weights_root()
    try:
        rel = source_path.relative_to(default_root)
    except ValueError:
        return weights_root / source_path.name
    return weights_root / rel


def _download_url(url: str, output_path: Path) -> Path:
    logger.info("Downloading %s", url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    logger.info("Saved file to %s", output_path)
    return output_path


def _pull_registry_model(spec_key: str, weights_root: Path) -> PullResult:
    spec = get_model_spec(spec_key)

    if spec.key in {"grabcut", "rembg"}:
        return PullResult(key=spec.key, status="skipped", files=[], skipped_reason="no downloadable remote assets")

    if spec.key == "birefnet":
        return PullResult(
            key=spec.key,
            status="skipped",
            files=[],
            skipped_reason="BiRefNet still loads on demand from Hugging Face and does not yet have a dedicated mirror path",
        )

    local_weights = spec.local_weights
    if local_weights is None:
        return PullResult(key=spec.key, status="skipped", files=[], skipped_reason="no local_weights path configured")

    target_root = _relocate_path(local_weights, weights_root)
    pulled_files: list[str] = []

    source_repo = spec.metadata.get("source_repo")
    source_onnx = spec.metadata.get("source_onnx")
    source_revision = str(spec.metadata.get("source_revision", "main"))
    if source_repo and source_onnx:
        onnx_source = Path(str(spec.metadata.get("onnx_path", "")))
        output_path = _relocate_path(onnx_source, weights_root) if str(onnx_source) else target_root / "onnx" / Path(str(source_onnx)).name
        download_hf_onnx(
            HFOnnxDownloadConfig(
                output_path=output_path,
                repo_id=str(source_repo),
                filename=str(source_onnx),
                revision=source_revision,
            )
        )
        pulled_files.append(str(output_path))

        source_processor = spec.metadata.get("source_processor")
        if source_processor:
            processor_source = Path(str(spec.metadata.get("processor_path", "")))
            processor_path = _relocate_path(processor_source, weights_root) if str(processor_source) else target_root / Path(str(source_processor)).name
            processor_path.parent.mkdir(parents=True, exist_ok=True)
            cached = _download_hf_file(str(source_repo), str(source_processor), source_revision)
            processor_path.write_bytes(cached.read_bytes())
            pulled_files.append(str(processor_path))

        return PullResult(key=spec.key, status="ok", files=pulled_files)

    return PullResult(
        key=spec.key,
        status="skipped",
        files=[],
        skipped_reason="no explicit source_repo/source_onnx metadata in the registry",
    )


def _pull_upscaler_model(spec_key: str, weights_root: Path) -> PullResult:
    from ..upscale.registry import get_upscaler_spec

    spec = get_upscaler_spec(spec_key)
    local_weights = spec.local_weights
    if local_weights is None:
        return PullResult(key=spec.key, status="skipped", files=[], skipped_reason="no local_weights path configured")

    source_url = spec.metadata.get("source_url")
    if not source_url:
        return PullResult(
            key=spec.key,
            status="skipped",
            files=[],
            skipped_reason="no explicit source_url metadata in the upscaler registry",
        )

    target_root = _relocate_path(local_weights, weights_root)
    output_path = target_root / str(spec.metadata.get("model_filename", Path(str(source_url)).name))
    _download_url(str(source_url), output_path)
    return PullResult(key=spec.key, status="ok", files=[str(output_path)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pull all registry-backed model assets into the local weights tree")
    parser.add_argument(
        "--weights-root",
        default=None,
        help="Optional override for the local weights root directory",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs to in addition to stderr")
    parser.add_argument("--log-json", action="store_true", help="Emit JSON log records instead of human-readable log lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, json_output=args.log_json, force=True)

    weights_root = Path(args.weights_root) if args.weights_root else _default_weights_root()
    weights_root.mkdir(parents=True, exist_ok=True)

    results: list[PullResult] = []
    for spec in list_model_specs():
        logger.info("pull start model=%s", spec.key)
        try:
            result = _pull_registry_model(spec.key, weights_root)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("pull failed model=%s", spec.key)
            result = PullResult(key=spec.key, status="failed", files=[], error=error)
        results.append(result)
        logger.info("pull result %s", asdict(result))

    for spec in list_upscaler_specs():
        logger.info("pull start upscaler=%s", spec.key)
        try:
            result = _pull_upscaler_model(spec.key, weights_root)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("pull failed upscaler=%s", spec.key)
            result = PullResult(key=spec.key, status="failed", files=[], error=error)
        results.append(result)
        logger.info("pull result %s", asdict(result))

    ok = sum(1 for result in results if result.status == "ok")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = sum(1 for result in results if result.status == "failed")
    logger.info("pull complete ok=%d skipped=%d failed=%d", ok, skipped, failed)
    if failed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
