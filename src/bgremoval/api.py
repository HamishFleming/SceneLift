from __future__ import annotations

from .logging_controller import get_logger, setup_logging
from .io import is_image_output, parse_source
from .methods import create_remover
from .live import LiveConfig, run_live_virtualcam
from .pipeline import RunConfig, run_image_file, run_video_or_camera


logger = get_logger(__name__)


def run(
    input_source: str,
    output: str,
    method: str = "grabcut",
    background_color: tuple[int, int, int] = (0, 0, 0),
    max_frames: int | None = None,
    virtualcam_device: str | None = None,
    live: bool | None = None,
    live_max_dimension: int | None = 1280,
    live_target_fps: float | None = None,
    log_level: int | str | None = None,
    log_file: str | None = None,
    log_json: bool = False,
) -> None:
    """Run background removal from Python code.

    Args:
        input_source: File path, camera spec like ``camera:0``, or device index string.
        output: Output file path or ``virtualcam``.
        method: One of ``grabcut``, ``rembg``, ``birefnet``, ``u2net-human-seg``, or ``mediapipe-selfie-segmentation``.
        background_color: RGB tuple used for non-transparent outputs.
        max_frames: Optional frame limit for camera/video inputs.
        virtualcam_device: Optional device name passed to ``pyvirtualcam``.
        live: Force the low-latency live pipeline.
        live_max_dimension: Optional max width/height used to downscale live frames.
        live_target_fps: Optional cap for the live output fps.
        log_level: Optional logging level to configure before running.
        log_file: Optional path to append logs to.
        log_json: Emit JSON log records instead of human-readable text.
    """

    if log_level is not None:
        setup_logging(log_level, log_file=log_file, json_output=log_json, force=True)
    elif log_file is not None or log_json:
        setup_logging("INFO", log_file=log_file, json_output=log_json, force=True)

    logger.info(
        "Starting run input=%s output=%s method=%s max_frames=%s",
        input_source,
        output,
        method,
        max_frames,
    )

    source = parse_source(input_source)
    remover = create_remover(method)
    logger.info("healthcheck component=backend status=ok method=%s loader=%s", method, type(remover).__name__)
    logger.info("healthcheck component=input status=ok kind=%s source=%s", source.kind, source.value)

    if source.kind == "image":
        if output == "virtualcam":
            raise ValueError("virtualcam output is only supported for video or camera input")
        if not is_image_output(output):
            raise ValueError("image input requires an image output path such as .png or .jpg")
        run_image_file(
            input_path=str(source.value),
            output_path=output,
            remover=remover,
            background_color=background_color,
        )
        logger.info("Completed image run for %s -> %s", input_source, output)
        return

    if source.kind not in {"video", "camera"}:
        raise ValueError(f"Unsupported input type: {source.kind}")
    if output != "virtualcam" and is_image_output(output):
        raise ValueError("video or camera input requires a video file output or virtualcam")

    use_live = live if live is not None else (source.kind == "camera" and output == "virtualcam")
    if use_live and output != "virtualcam":
        raise ValueError("live mode currently targets virtualcam output")

    if use_live:
        logger.info("Using live low-latency pipeline")
        run_live_virtualcam(
            LiveConfig(
                input_source=source,
                method=remover,
                background_color=background_color,
                virtualcam_device=virtualcam_device,
                max_frames=max_frames,
                max_dimension=live_max_dimension,
                target_fps=live_target_fps,
            )
        )
        logger.info("Completed live run for %s -> %s", input_source, output)
        return

    run_video_or_camera(
        RunConfig(
            input_source=source,
            output=output,
            method=remover,
            background_color=background_color,
            max_frames=max_frames,
            virtualcam_device=virtualcam_device,
        )
    )
    logger.info("Completed stream run for %s -> %s", input_source, output)
