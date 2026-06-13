import sys
from pathlib import Path

from loguru import logger


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        output_dir / "train.log",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        enqueue=True,
    )
