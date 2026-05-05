import logging
import sys
from pathlib import Path


def setup_logging(
    *,
    level: str = "INFO",
    log_file: Path | None = None,
    disable_logging: bool = False,
) -> None:
    if disable_logging:
        logging.disable(logging.CRITICAL)
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
