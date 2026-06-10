from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    value = level.strip().upper()
    if value.isdigit():
        return int(value)
    resolved = logging.getLevelName(value)
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"Unknown log level: {level}")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=True)


@dataclass(frozen=True)
class LoggingOptions:
    level: int | str = logging.INFO
    log_file: str | None = None
    json_output: bool = False
    log_format: str = DEFAULT_LOG_FORMAT


class LoggingController:
    _configured = False

    @classmethod
    def configure(
        cls,
        level: int | str = logging.INFO,
        *,
        log_file: str | None = None,
        json_output: bool = False,
        log_format: str = DEFAULT_LOG_FORMAT,
        force: bool = False,
    ) -> None:
        resolved_level = _coerce_level(level)
        root = logging.getLogger()
        if force or not cls._configured:
            for handler in list(root.handlers):
                root.removeHandler(handler)

            root.setLevel(resolved_level)
            formatter: logging.Formatter = JsonFormatter() if json_output else logging.Formatter(log_format)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(resolved_level)
            stream_handler.setFormatter(formatter)
            root.addHandler(stream_handler)

            if log_file:
                path = Path(log_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(path, encoding="utf-8")
                file_handler.setLevel(resolved_level)
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)

            cls._configured = True
        else:
            root.setLevel(resolved_level)

    @classmethod
    def get_logger(cls, name: str | None = None) -> logging.Logger:
        return logging.getLogger(name)


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | None = None,
    json_output: bool = False,
    log_format: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
) -> None:
    LoggingController.configure(
        level=level,
        log_file=log_file,
        json_output=json_output,
        log_format=log_format,
        force=force,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return LoggingController.get_logger(name)
